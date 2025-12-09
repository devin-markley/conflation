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

std::vector<MatchResult> GreedyMatchChunk(
    const std::vector<OGRFeature*>& allTdaFeatures,
    const std::vector<OGRFeature*>& allOsmFeatures,
    int limit,
    int rank,
    int size)
{
    int n = std::min(limit, (int)allTdaFeatures.size());

    // Simpler workload calculation
    int baseChunk = n / size;
    int remainder = n % size;
    int startIdx = rank * baseChunk + std::min(rank, remainder);
    int localSize = baseChunk + (rank < remainder ? 1 : 0);
    int endIdx = startIdx + localSize;

    printf("Rank %d processing TDA features [%d, %d) = %d features\n",
           rank, startIdx, endIdx, localSize);

    std::vector<MatchResult> results(localSize);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < localSize; ++i)
    {
        OGRFeature* tda = allTdaFeatures[startIdx + i];

        double minDist = std::numeric_limits<double>::max();
        long bestOsmFID = -1;

        for (size_t j = 0; j < allOsmFeatures.size(); ++j)
        {
            double dist = FeatureHausdorffDistance(tda, allOsmFeatures[j]);
            if (dist < minDist)
            {
                minDist = dist;
                bestOsmFID = allOsmFeatures[j]->GetFID();
            }
        }

        results[i] = {tda->GetFID(), bestOsmFID, minDist};
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

    // Gather counts from all ranks (all ranks need to know counts for displacements)
    std::vector<int> allCounts(size);
    MPI_Allgather(&localCount, 1, MPI_INT,
                  allCounts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    // Calculate displacements and total count
    std::vector<int> displacements(size);
    int totalCount = 0;
    for (int i = 0; i < size; ++i) {
        displacements[i] = totalCount;
        totalCount += allCounts[i];
    }

    // Convert counts and displacements to bytes (since we're using MPI_BYTE)
    std::vector<int> byteCounts(size);
    std::vector<int> byteDisplacements(size);
    for (int i = 0; i < size; ++i) {
        byteCounts[i] = allCounts[i] * sizeof(MatchResult);
        byteDisplacements[i] = displacements[i] * sizeof(MatchResult);
    }

    // Allocate receive buffer on rank 0
    std::vector<MatchResult> allResults(rank == 0 ? totalCount : 0);

    // Gather all results to rank 0
    MPI_Gatherv(localResults.data(),
                localCount * sizeof(MatchResult),
                MPI_BYTE,
                allResults.data(),
                byteCounts.data(),
                byteDisplacements.data(),
                MPI_BYTE,
                0,
                MPI_COMM_WORLD);

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

    if (rank == 0) {
        printf("\nFirst 5 TDA features:\n");
        for (int i = 0; i < 5 && i < floridaTDA.size(); ++i)
            PrintFeature(floridaTDA[i]);

        printf("\nFirst 5 OSM features:\n");
        for (int i = 0; i < 5 && i < floridaOSM.size(); ++i)
            PrintFeature(floridaOSM[i]);
    }

    int match_limit = 100;

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Each rank processes its chunk
    std::vector<MatchResult> localResults = GreedyMatchChunk(
        floridaTDA, floridaOSM, match_limit, rank, size);

    // Gather all results to rank 0
    std::vector<MatchResult> allResults = GatherResults(localResults, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    //  rank 0 reconstructs and prints matches
    if (rank == 0) {
        std::vector<Match> matches = ReconstructMatches(
            allResults, floridaTDA, floridaOSM);
        printf("\nMatching complete. Total successful matches: %zu\n", matches.size());
        PrintMatches(matches);
        printf("Matching took %f seconds\n", endTime - startTime);
    }

    for (auto f : floridaTDA) OGRFeature::DestroyFeature(f);
    for (auto f : floridaOSM) OGRFeature::DestroyFeature(f);

    MPI_Finalize();
    return 0;
}