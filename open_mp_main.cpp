#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <ogrsf_frmts.h>
#include <omp.h>
#include "common.h"



std::vector<Match> GreedyMatch(const std::vector<OGRFeature*>& tdaFeatures,
                               const std::vector<OGRFeature*>& osmFeatures,
                               int limit)
{
    int n = std::min(limit, (int)tdaFeatures.size());
    std::vector<Match> matches(n);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; ++i)
    {
        OGRFeature* tda = tdaFeatures[i];

        double minDist = std::numeric_limits<double>::max();
        OGRFeature* bestOSM = nullptr;

        for (int j = 0; j < osmFeatures.size(); ++j)
        {
            double dist = FeatureHausdorffDistance(tda, osmFeatures[j]);
            if (dist < minDist)
            {
                minDist = dist;
                bestOSM = osmFeatures[j];
            }
        }

        matches[i] = Match{ tda, bestOSM, minDist };
    }

    return matches;
}

int main()
{
    omp_set_num_threads(16);
    GDALAllRegister();

    printf("\nReading TDA features (Speed Limits)...\n");
    std::vector<OGRFeature*> floridaTDA = ReadFeatures(PATH_TO_FLORIDA_MAX_SPEED_LIMIT, "Maximum_Speed_Limit_TDA");

    printf("\nReading OSM data (Roads, filtered)...\n");
    std::vector<OGRFeature*> floridaOSM = ReadFeatures(PATH_TO_FLORIDA_OSM, "lines", "highway IS NOT NULL");

    printf("\nFirst 5 TDA features:\n");
    for (int i = 0; i < 5 && i < floridaTDA.size(); ++i)
        PrintFeature(floridaTDA[i]);

    printf("\nFirst 5 OSM features:\n");
    for (int i = 0; i < 5 && i < floridaOSM.size(); ++i)
        PrintFeature(floridaOSM[i]);

    int match_limit = 100;

    double startTime = omp_get_wtime();
    std::vector<Match> matches = GreedyMatch(floridaTDA, floridaOSM, match_limit);
    double endTime = omp_get_wtime();

    printf("\nMatching complete. Total successful matches: %zu\n", matches.size());
    PrintMatches(matches);

    printf("Matching took %f seconds\n", endTime - startTime);

    for (auto f : floridaTDA) OGRFeature::DestroyFeature(f);
    for (auto f : floridaOSM) OGRFeature::DestroyFeature(f);

    return 0;
}
