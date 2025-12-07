#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <ogrsf_frmts.h>
#include <omp.h>

const char* PATH_TO_FLORIDA_MAX_SPEED_LIMIT =
    "/Users/devinmarkley/CLionProjects/untitled3/MSL_Florida.geojson";
const char* PATH_TO_FLORIDA_OSM =
    "/Users/devinmarkley/CLionProjects/untitled3/florida.osm.pbf";

struct Match
{
    OGRFeature* tda_feature;
    OGRFeature* osm_feature;
    double distance;
};

// Compute the straight-line distance between two points
double PointDistance(const OGRPoint& p1, const OGRPoint& p2)
{
    double dx = p1.getX() - p2.getX();
    double dy = p1.getY() - p2.getY();
    return std::sqrt(dx * dx + dy * dy);
}

// Extract all coordinate points from a LineString geometry
std::vector<OGRPoint> GetPoints(OGRGeometry* geom)
{
    std::vector<OGRPoint> points;
    if (!geom) return points;

    OGRLineString* line = (OGRLineString*)geom;
    for (int i = 0; i < line->getNumPoints(); ++i)
    {
        OGRPoint p;
        line->getPoint(i, &p);
        points.push_back(p);
    }
    return points;
}

// Directed Hausdorff distance (from points in geom1 to nearest points in geom2)
double DirectedHausdorff(OGRGeometry* geom1, OGRGeometry* geom2)
{
    double max_min_dist = 0.0;
    std::vector<OGRPoint> pts1 = GetPoints(geom1);
    std::vector<OGRPoint> pts2 = GetPoints(geom2);

    if (pts1.empty() || pts2.empty()) return std::numeric_limits<double>::max();

    for (auto& p1 : pts1)
    {
        double min_dist_to_p2 = std::numeric_limits<double>::max();

        for (auto& p2 : pts2)
        {
            double d = PointDistance(p1, p2);
            min_dist_to_p2 = std::min(min_dist_to_p2, d);
        }
        max_min_dist = std::max(max_min_dist, min_dist_to_p2);
    }
    return max_min_dist;
}

// Symmetric Hausdorff distance (the maximum of the two directed distances)
double FeatureHausdorffDistance(OGRFeature* f1, OGRFeature* f2)
{
    OGRGeometry* g1 = f1->GetGeometryRef();
    OGRGeometry* g2 = f2->GetGeometryRef();

    if (!g1 || !g2) return std::numeric_limits<double>::max();

    return std::max(DirectedHausdorff(g1, g2),
                    DirectedHausdorff(g2, g1));
}

void PrintGeometry(OGRFeature* f)
{
    OGRGeometry *poGeom = f->GetGeometryRef();
    if (poGeom)
    {
        char *wkt = nullptr;
        poGeom->exportToWkt(&wkt);
        printf("%s", (wkt));
        CPLFree(wkt);
    } else {
        printf(" Geometry: (null)");
    }
    printf("\n");
}

void PrintFeature(OGRFeature* f)
{
    for (int i = 0; i < f->GetFieldCount(); ++i)
    {
        const char* field_value = f->IsFieldSetAndNotNull(i) ? f->GetFieldAsString(i) : "(null)";
        printf("%s,", field_value);
    }

    PrintGeometry(f);
}

void PrintMatches(const std::vector<Match>& matches)
{
    for (size_t i = 0; i < 6; ++i)
    {
        OGRFeature* tda = matches[i].tda_feature;
        OGRFeature* osm = matches[i].osm_feature;

        // --- Use GetFieldIndex() to safely retrieve fields by name ---

        // 1. Get the field index for TDA speed
        int tda_speed_index = tda->GetFieldIndex("speed");

        // 2. Get the field index for OSM speed
        int osm_speed_index = osm->GetFieldIndex("max_speed");


        // --- Retrieve Speed Limits ---

        // Check if the field index is valid (not -1) AND the field is set
        const char* tda_speed = (tda_speed_index != -1 && tda->IsFieldSetAndNotNull(tda_speed_index))
                                ? tda->GetFieldAsString(tda_speed_index)
                                : "(N/A)";

        const char* osm_speed = (osm_speed_index != -1 && osm->IsFieldSetAndNotNull(osm_speed_index))
                                ? osm->GetFieldAsString(osm_speed_index)
                                : "(N/A)";

        printf("Distance: %.6f\n", i + 1, matches[i].distance);

        printf("  TDA Speed Limit: %s\n", tda_speed);
        printf("  OSM Speed Limit: %s\n", osm_speed);

        printf("TDA: ");
        PrintGeometry(tda);
        printf("\n");
        printf("OSM: ");
        PrintGeometry(osm);

        printf("----------------------------------\n");
    }
}

std::vector<OGRFeature*> ReadFeatures(const char* path, const char* layerName, const char* filter = nullptr)
{
    std::vector<OGRFeature*> features;

    GDALDataset *poDS = (GDALDataset*) GDALOpenEx(
        path, GDAL_OF_VECTOR, NULL, NULL, NULL);

    if (!poDS) {
        fprintf(stderr, "Error: Failed to open dataset at %s\n", path);
        return features;
    }

    OGRLayer *poLayer = poDS->GetLayerByName(layerName);
    if (!poLayer) {
        fprintf(stderr, "Error: Layer '%s' not found!\n", layerName);
        GDALClose(poDS);
        return features;
    }

    if (filter)
        poLayer->SetAttributeFilter(filter);

    poLayer->ResetReading();
    OGRFeature *poFeature;
    while ((poFeature = poLayer->GetNextFeature()) != nullptr)
        features.push_back(poFeature);

    printf("Total features read from %s: %zu\n", layerName, features.size());

    GDALClose(poDS);
    return features;
}

// Finds the closest OSM feature for a limited number of TDA features.
std::vector<Match> GreedyMatch(const std::vector<OGRFeature*>& tdaFeatures,
                               std::vector<OGRFeature*>& osmFeatures,
                               int limit)
{
    std::vector<Match> matches;

    // Iterate through TDA features up to the limit
    for (int i = 0; i < limit && i < tdaFeatures.size(); ++i)
    {
        OGRFeature* current_tda_feature = tdaFeatures[i];

        double minDist = std::numeric_limits<double>::max();
        OGRFeature* closestOSM = nullptr;

        // Iterator that points to the closest OSM feature found
        auto it_closest_osm = osmFeatures.end();

        // Inner loop: Iterate through all available OSM features
        for (auto it = osmFeatures.begin(); it != osmFeatures.end(); ++it)
        {
            OGRFeature* current_osm_feature = *it;
            double dist = FeatureHausdorffDistance(current_tda_feature, current_osm_feature);

            if (dist < minDist)
            {
                minDist = dist;
                closestOSM = current_osm_feature;
                it_closest_osm = it;
            }
        }

        // If a valid match was found
        if (closestOSM)
        {
            matches.push_back(Match{ current_tda_feature, closestOSM, minDist});

            // Remove the matched OSM feature to ensure a 1:1 match.
            osmFeatures.erase(it_closest_osm);
        }
    }

    return matches;
}

int main()
{
    omp_set_num_threads(16);
    GDALAllRegister();

    printf("\nReading TDA features (Speed Limits)...\n");
    std::vector<OGRFeature*> floridaTDA = ReadFeatures(
        PATH_TO_FLORIDA_MAX_SPEED_LIMIT, "Maximum_Speed_Limit_TDA");

    printf("\nReading OSM data (Roads, filtered)...\n");
    std::vector<OGRFeature*> floridaOSM = ReadFeatures(
        PATH_TO_FLORIDA_OSM, "lines", "highway IS NOT NULL");

    printf("\nFirst 5 TDA features:\n");
    for (int i = 0; i < 5 && i < floridaTDA.size(); ++i)
        PrintFeature(floridaTDA[i]);

    printf("\nFirst 5 OSM features:\n");
    for (int i = 0; i < 5 && i < floridaOSM.size(); ++i)
        PrintFeature(floridaOSM[i]);

    int match_limit = 10;

    double startTime = omp_get_wtime();
    std::vector<Match> matches = GreedyMatch(floridaTDA, floridaOSM, match_limit);
    double endTime = omp_get_wtime();

    printf("\nMatching complete. Total successful matches: %zu\n", matches.size());
    PrintMatches(matches);

    printf("Matching took %f seconds\n", endTime - startTime);

    for (OGRFeature* f : floridaTDA)
        OGRFeature::DestroyFeature(f);

    for (OGRFeature* f : floridaOSM)
        OGRFeature::DestroyFeature(f);

    return 0;
}