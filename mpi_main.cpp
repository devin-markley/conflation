#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <mpi.h>
#include <omp.h>
#include <ogrsf_frmts.h>
#include "common.h"

struct MatchResult {
    long tdaFID;
    long osmFID;
    double distance;
};

std::vector<MatchResult> ProcessLocalChunk(
    const std::vector<OGRFeature*>& allTdaFeatures,
    const std::vector<OGRFeature*>& allOsmFeatures,
    int limit,
    int rank,
    int size)
{
    int n = std::min(limit, (int)allTdaFeatures.size());

    // Calculate this rank's workload
    int chunkSize = n / size;
    int remainder = n % size;
    int startIdx = rank * chunkSize + std::min(rank, remainder);
    int endIdx = startIdx + chunkSize + (rank < remainder ? 1 : 0);
    int localSize = endIdx - startIdx;

    printf("Rank %d processing TDA features [%d, %d) = %d features\n",
           rank, startIdx, endIdx, localSize);

    std::vector<MatchResult> results(localSize);

    // Process with OpenMP
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < localSize; ++i)
    {
        int globalIdx = startIdx + i;
        OGRFeature* tda = allTdaFeatures[globalIdx];

        double minDist = std::numeric_limits<double>::max();
        long bestOsmFID = -1;

        // Search all OSM features
        for (size_t j = 0; j < allOsmFeatures.size(); ++j)
        {
            double dist = FeatureHausdorffDistance(tda, allOsmFeatures[j]);
            if (dist < minDist)
            {
                minDist = dist;
                bestOsmFID = allOsmFeatures[j]->GetFID();
            }
        }

        results[i] = MatchResult{
            tda->GetFID(),
            bestOsmFID,
            minDist
        };
    }

    return results;
}

// Gather results from all ranks to rank 0
std::vector<MatchResult> GatherResults(
    const std::vector<MatchResult>& localResults,
    int rank,
    int size)
{
    int localCount = localResults.size();
    std::vector<int> recvCounts(size);

    // Gather counts
    MPI_Gather(&localCount, 1, MPI_INT,
               recvCounts.data(), 1, MPI_INT,
               0, MPI_COMM_WORLD);

    std::vector<MatchResult> allResults;

    if (rank == 0) {
        // Calculate displacements
        int totalCount = 0;
        std::vector<int> displs(size);
        displs[0] = 0;

        for (int i = 0; i < size; ++i) {
            totalCount += recvCounts[i];
            if (i > 0) {
                displs[i] = displs[i-1] + recvCounts[i-1];
            }
            // Convert to bytes
            recvCounts[i] *= sizeof(MatchResult);
            displs[i] *= sizeof(MatchResult);
        }

        allResults.resize(totalCount);

        // Gather all results
        int localBytes = localCount * sizeof(MatchResult);
        MPI_Gatherv(localResults.data(), localBytes, MPI_BYTE,
                    allResults.data(), recvCounts.data(), displs.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);
    } else {
        // Non-root ranks just send
        int localBytes = localCount * sizeof(MatchResult);
        MPI_Gatherv(localResults.data(), localBytes, MPI_BYTE,
                    nullptr, nullptr, nullptr, MPI_BYTE,
                    0, MPI_COMM_WORLD);
    }

    return allResults;
}

// Reconstruct Match objects
std::vector<Match> ReconstructMatches(
    const std::vector<MatchResult>& results,
    const std::vector<OGRFeature*>& tdaFeatures,
    const std::vector<OGRFeature*>& osmFeatures)
{
    std::vector<Match> matches(results.size());

    for (size_t i = 0; i < results.size(); ++i) {
        OGRFeature* tda = nullptr;
        for (auto f : tdaFeatures) {
            if (f->GetFID() == results[i].tdaFID) {
                tda = f;
                break;
            }
        }

        OGRFeature* osm = nullptr;
        if (results[i].osmFID != -1) {
            for (auto f : osmFeatures) {
                if (f->GetFID() == results[i].osmFID) {
                    osm = f;
                    break;
                }
            }
        }

        matches[i] = Match{ tda, osm, results[i].distance };
    }

    return matches;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    omp_set_num_threads(8);

    GDALAllRegister();

    // All ranks read the data independently
    if (rank == 0) printf("\nAll ranks reading TDA features (Speed Limits)...\n");
    std::vector<OGRFeature*> floridaTDA = ReadFeatures(
        PATH_TO_FLORIDA_MAX_SPEED_LIMIT, "Maximum_Speed_Limit_TDA");

    if (rank == 0) printf("\nAll ranks reading OSM data (Roads, filtered)...\n");
    std::vector<OGRFeature*> floridaOSM = ReadFeatures(
        PATH_TO_FLORIDA_OSM, "lines", "highway IS NOT NULL");

    // Only rank 0 prints sample data
    if (rank == 0) {
        printf("\nFirst 5 TDA features:\n");
        for (int i = 0; i < 5 && i < floridaTDA.size(); ++i)
            PrintFeature(floridaTDA[i]);

        printf("\nFirst 5 OSM features:\n");
        for (int i = 0; i < 5 && i < floridaOSM.size(); ++i)
            PrintFeature(floridaOSM[i]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int match_limit = 100;

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Each rank processes its chunk
    std::vector<MatchResult> localResults = ProcessLocalChunk(
        floridaTDA, floridaOSM, match_limit, rank, size);

    // Gather all results to rank 0
    std::vector<MatchResult> allResults = GatherResults(localResults, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    //  rank 0 reconstructs and prints
    if (rank == 0) {
        std::vector<Match> matches = ReconstructMatches(
            allResults, floridaTDA, floridaOSM);
        PrintMatches(matches);
    }

    // Cleanup
    for (auto f : floridaTDA) OGRFeature::DestroyFeature(f);
    for (auto f : floridaOSM) OGRFeature::DestroyFeature(f);

    MPI_Finalize();
    return 0;
}