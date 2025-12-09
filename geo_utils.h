#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <ogrsf_frmts.h>

struct Match
{
    OGRFeature* tda_feature;
    OGRFeature* osm_feature;
    double distance;
};

// Extract all coordinate points from a LineString geometry
inline std::vector<OGRPoint> GetPoints(OGRGeometry* geom)
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

// Great-circle distance (Haversine) in meters
inline double Haversine(const OGRPoint& p1, const OGRPoint& p2)
{
    static const double R = 6371000.0; // Earth radius in meters

    double lat1 = p1.getY() * M_PI / 180.0;
    double lon1 = p1.getX() * M_PI / 180.0;
    double lat2 = p2.getY() * M_PI / 180.0;
    double lon2 = p2.getX() * M_PI / 180.0;

    double dlat = lat2 - lat1;
    double dlon = lon2 - lon1;

    double a = sin(dlat / 2) * sin(dlat / 2) +
               cos(lat1) * cos(lat2) *
               sin(dlon / 2) * sin(dlon / 2);

    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    return R * c;
}

// Directed Hausdorff distance
inline double DirectedHausdorff(OGRGeometry* geom1, OGRGeometry* geom2)
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
            double d = Haversine(p1, p2);
            min_dist_to_p2 = std::min(min_dist_to_p2, d);
        }
        max_min_dist = std::max(max_min_dist, min_dist_to_p2);
    }
    return max_min_dist;
}

// Symmetric Hausdorff distance
inline double FeatureHausdorffDistance(OGRFeature* f1, OGRFeature* f2)
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
    }
    else
    {
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
    for (size_t i = 0; i < 6 && i < matches.size(); ++i)
    {
        printf("Distance: %.6f\n", matches[i].distance);
        printf("TDA: ");
        PrintGeometry(matches[i].tda_feature);
        printf("\nOSM: ");
        PrintFeature(matches[i].osm_feature);
        printf("----------------------------------\n");
    }
}

std::vector<OGRFeature*> ReadFeatures(const char* path, const char* layerName, const char* filter = nullptr)
{
    std::vector<OGRFeature*> features;

    GDALDataset *poDS = (GDALDataset*) GDALOpenEx(
        path, GDAL_OF_VECTOR, NULL, NULL, NULL);

    if (!poDS)
    {
        fprintf(stderr, "Error: Failed to open dataset at %s\n", path);
        return features;
    }

    OGRLayer *poLayer = poDS->GetLayerByName(layerName);
    if (!poLayer)
    {
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