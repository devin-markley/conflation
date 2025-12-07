#include <iostream>
#include <vector>
#include <ogrsf_frmts.h>
#include <omp.h>
#include <cmath>
#include <limits>

const char* PATH_TO_FLORIDA_MAX_SPEED_LIMIT =
    "/Users/devinmarkley/CLionProjects/untitled3/MSL_Florida.geojson";
const char* PATH_TO_FLORIDA_OSM =
    "/Users/devinmarkley/CLionProjects/untitled3/florida.osm.pbf";

struct Match
{
    OGRFeature* tda;
    OGRFeature* osm;
    double distance;
};

void PrintFeature(OGRFeature* f)
{
    for (int i = 0; i < f->GetFieldCount(); ++i)
    {
        if (f->IsFieldSetAndNotNull(i))
            std::cout << f->GetFieldAsString(i);
        else
            std::cout << "(null)";
        std::cout << ",";
    }

    OGRGeometry *poGeom = f->GetGeometryRef();
    if(poGeom)
    {
        char *wkt = nullptr;
        poGeom->exportToWkt(&wkt);
        std::cout << wkt;
        CPLFree(wkt);
    }

    std::cout << std::endl;
}

void PrintMatches(const std::vector<Match>& matches)
{
    for (size_t i = 0; i < matches.size(); ++i)
    {
        std::cout << "Match " << i + 1 << ", Distance: " << matches[i].distance << std::endl;
        std::cout << "TDA: ";
        PrintFeature(matches[i].tda);
        std::cout << "OSM: ";
        PrintFeature(matches[i].osm);
        std::cout << "----------------------------------" << std::endl;
    }
}

// Helper: compute Euclidean distance between two points
double PointDistance(const OGRPoint& p1, const OGRPoint& p2)
{
    double dx = p1.getX() - p2.getX();
    double dy = p1.getY() - p2.getY();
    return std::sqrt(dx*dx + dy*dy);
}

// Extract points from LineString geometry
std::vector<OGRPoint> GetPoints(OGRGeometry* geom)
{
    std::vector<OGRPoint> points;
    if (!geom) return points;

    OGRwkbGeometryType type = wkbFlatten(geom->getGeometryType());
    if (type == wkbLineString)
    {
        OGRLineString* line = geom->toLineString();
        for (int i = 0; i < line->getNumPoints(); ++i)
        {
            OGRPoint p;
            line->getPoint(i, &p);
            points.push_back(p);
        }
    }
    else if (type == wkbPolygon)
    {
        OGRPolygon* poly = geom->toPolygon();
        OGRLinearRing* ring = poly->getExteriorRing();
        for (int i = 0; i < ring->getNumPoints(); ++i)
        {
            OGRPoint p;
            ring->getPoint(i, &p);
            points.push_back(p);
        }
    }
    // Could add more types if needed
    return points;
}

// Directed Hausdorff distance (from geom1 to geom2)
double DirectedHausdorff(OGRGeometry* geom1, OGRGeometry* geom2)
{
    double maxMinDist = 0.0;
    std::vector<OGRPoint> pts1 = GetPoints(geom1);
    std::vector<OGRPoint> pts2 = GetPoints(geom2);

    if (pts1.empty() || pts2.empty()) return std::numeric_limits<double>::max();

    for (auto& p1 : pts1)
    {
        double minDist = std::numeric_limits<double>::max();
        for (auto& p2 : pts2)
        {
            double d = PointDistance(p1, p2);
            if (d < minDist) minDist = d;
        }
        if (minDist > maxMinDist) maxMinDist = minDist;
    }
    return maxMinDist;
}

// Symmetric Hausdorff distance
double FeatureHausdorffDistance(OGRFeature* f1, OGRFeature* f2)
{
    OGRGeometry* g1 = f1->GetGeometryRef();
    OGRGeometry* g2 = f2->GetGeometryRef();

    if (!g1 || !g2) return std::numeric_limits<double>::max();

    double dist = std::max(DirectedHausdorff(g1, g2),
                           DirectedHausdorff(g2, g1));

    return dist;
}

// --------------------- Reading datasets ---------------------

std::vector<OGRFeature*> ReadFloridaTDA()
{
    std::vector<OGRFeature*> features;

    GDALDataset *poDS = (GDALDataset*) GDALOpenEx(
        PATH_TO_FLORIDA_MAX_SPEED_LIMIT, GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (!poDS)
    {
        std::cerr << "Failed to open Florida TDA dataset." << std::endl;
        return features;
    }

    OGRLayer *poLayer = poDS->GetLayerByName("Maximum_Speed_Limit_TDA");
    if (!poLayer)
    {
        std::cerr << "Layer not found!" << std::endl;
        GDALClose(poDS);
        return features;
    }

    poLayer->ResetReading();
    OGRFeature *poFeature;
    while ((poFeature = poLayer->GetNextFeature()) != nullptr)
        features.push_back(poFeature);

    std::cout << "Total Florida TDA features: " << features.size() << std::endl;

    GDALClose(poDS);
    return features;
}

std::vector<OGRFeature*> ReadFloridaOSM()
{
    std::vector<OGRFeature*> features;

    GDALDataset *poDS = (GDALDataset*) GDALOpenEx(
        PATH_TO_FLORIDA_OSM, GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!poDS)
    {
        std::cerr << "Failed to open Florida OSM dataset." << std::endl;
        return features;
    }

    OGRLayer *poLayer = poDS->GetLayerByName("lines");
    if (!poLayer)
    {
        std::cerr << "Lines layer not found!" << std::endl;
        GDALClose(poDS);
        return features;
    }

    poLayer->SetAttributeFilter("highway IS NOT NULL");
    poLayer->ResetReading();

    OGRFeature *poFeature;
    while ((poFeature = poLayer->GetNextFeature()) != nullptr)
        features.push_back(poFeature);

    std::cout << "Total Florida OSM roads: " << features.size() << std::endl;

    GDALClose(poDS);
    return features;
}

// --------------------- Greedy matching ---------------------

std::vector<Match> GreedyMatch(const std::vector<OGRFeature*>& tdaFeatures,
                               std::vector<OGRFeature*>& osmFeatures)
{
    std::vector<Match> matches;

    size_t limit = std::min(tdaFeatures.size(), size_t(10)); // first 10 features
    for (size_t i = 0; i < limit; ++i)
    {
        auto tdaFeature = tdaFeatures[i];

        double minHausdorff = std::numeric_limits<double>::max();
        OGRFeature* closestOSM = nullptr;
        auto itClosest = osmFeatures.end();

        for (auto it = osmFeatures.begin(); it != osmFeatures.end(); ++it)
        {
            double dist = FeatureHausdorffDistance(tdaFeature, *it);
            if (dist < minHausdorff)
            {
                minHausdorff = dist;
                closestOSM = *it;
                itClosest = it;
            }
        }

        if (closestOSM)
        {
            matches.push_back(Match{ tdaFeature, closestOSM, minHausdorff });
            osmFeatures.erase(itClosest); // ensure 1:1
        }
    }

    return matches;
}

// --------------------- Main ---------------------

int main()
{
    omp_set_num_threads(16);
    GDALAllRegister();

    std::cout << "\nReading Florida TDA features:" << std::endl;
    std::vector<OGRFeature*> floridaTDA = ReadFloridaTDA();

    std::cout << "\nFirst 10 Florida TDA features:" << std::endl;
    for (int i = 0; i < 10 && i < floridaTDA.size(); ++i)
        PrintFeature(floridaTDA[i]);

    std::cout << "\nReading Florida OSM data:" << std::endl;
    std::vector<OGRFeature*> floridaOSM = ReadFloridaOSM();

    std::cout << "\nFirst 10 Florida OSM roads:" << std::endl;
    for (int i = 0; i < 10 && i < floridaOSM.size(); ++i)
        PrintFeature(floridaOSM[i]);

    double startTime = omp_get_wtime();
    std::vector<Match> matches = GreedyMatch(floridaTDA, floridaOSM);
    double endTime = omp_get_wtime();

    std::cout << "\nTotal matches: " << matches.size() << std::endl;
    PrintMatches(matches);

    printf("Matching took %f seconds\n", endTime - startTime);

    // Cleanup
    for (auto f : floridaTDA) OGRFeature::DestroyFeature(f);
    for (auto f : floridaOSM) OGRFeature::DestroyFeature(f);

    return 0;
}
