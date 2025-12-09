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

// Serializable structure for MPI communication
struct MatchData {
    long tdaFID;
    long osmFID;
    double distance;
};

std::vector<Match> HybridGreedyMatch(
    const std::vector<OGRFeature*>& tdaFeatures,
    const std::vector<OGRFeature*>& osmFeatures,
    int limit,
    int rank,
    int size)
{
    int n = std::min(limit, (int)tdaFeatures.size());

    // Calculate workload distribution for this MPI rank
    int chunkSize = n / size;
    int remainder = n % size;
    int startIdx = rank * chunkSize + std::min(rank, remainder);
    int endIdx = startIdx + chunkSize + (rank < remainder ? 1 : 0);
    int localSize = endIdx - startIdx;

    std::vector<Match> localMatches(localSize);

    if (rank == 0) {
        printf("Distributing %d TDA features across %d MPI ranks\n", n, size);
    }

    printf("Rank %d processing indices [%d, %d) - %d features\n",
           rank, startIdx, endIdx, localSize);

    // Each MPI rank processes its chunk using OpenMP
    #pragma omp parallel for schedule(dynamic, 1)
    for (int idx = 0; idx < localSize; ++idx)
    {
        int i = startIdx + idx;
        OGRFeature* tda = tdaFeatures[i];

        double minDist = std::numeric_limits<double>::max();
        OGRFeature* bestOSM = nullptr;

        // Inner loop: search all OSM features (read-only, thread-safe)
        for (size_t j = 0; j < osmFeatures.size(); ++j)
        {
            double dist = FeatureHausdorffDistance(tda, osmFeatures[j]);
            if (dist < minDist)
            {
                minDist = dist;
                bestOSM = osmFeatures[j];
            }
        }

        localMatches[idx] = Match{ tda, bestOSM, minDist };
    }

    return localMatches;
}

std::vector<Match> GatherMatches(
    const std::vector<Match>& localMatches,
    const std::vector<OGRFeature*>& tdaFeatures,
    const std::vector<OGRFeature*>& osmFeatures,
    int limit,
    int rank,
    int size)
{
    // Convert matches to serializable format
    std::vector<MatchData> localData(localMatches.size());
    for (size_t i = 0; i < localMatches.size(); ++i) {
        localData[i].tdaFID = localMatches[i].tda_feature->GetFID();
        localData[i].osmFID = localMatches[i].osm_feature->GetFID();
        localData[i].distance = localMatches[i].distance;
    }

    // Gather sizes from all ranks
    int localCount = localData.size();
    std::vector<int> recvCounts(size);
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Match> allMatches;

    if (rank == 0) {
        // Calculate displacements and adjust sizes for byte-based gather
        std::vector<int> displs(size);
        displs[0] = 0;
        for (int i = 0; i < size; ++i) {
            recvCounts[i] *= sizeof(MatchData);
            if (i > 0) {
                displs[i] = displs[i-1] + recvCounts[i-1];
            }
        }

        int totalBytes = displs[size-1] + recvCounts[size-1];
        int totalCount = totalBytes / sizeof(MatchData);
        std::vector<MatchData> allData(totalCount);

        // Gather all match data
        int localBytes = localCount * sizeof(MatchData);
        MPI_Gatherv(localData.data(), localBytes, MPI_BYTE,
                    allData.data(), recvCounts.data(), displs.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);

        // Reconstruct matches from serialized data
        allMatches.resize(totalCount);
        for (int i = 0; i < totalCount; ++i) {
            // Find features by FID
            OGRFeature* tda = nullptr;
            for (auto f : tdaFeatures) {
                if (f->GetFID() == allData[i].tdaFID) {
                    tda = f;
                    break;
                }
            }

            OGRFeature* osm = nullptr;
            if (allData[i].osmFID != -1) {
                for (auto f : osmFeatures) {
                    if (f->GetFID() == allData[i].osmFID) {
                        osm = f;
                        break;
                    }
                }
            }

            allMatches[i] = Match{ tda, osm, allData[i].distance };
        }
    } else {
        // Non-root ranks just send their data
        int localBytes = localCount * sizeof(MatchData);
        MPI_Gatherv(localData.data(), localBytes, MPI_BYTE,
                    nullptr, nullptr, nullptr, MPI_BYTE,
                    0, MPI_COMM_WORLD);
    }

    return allMatches;
}

int main(int argc, char** argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set OpenMP threads per MPI rank
    int threadsPerRank = 8;
    omp_set_num_threads(threadsPerRank);

    if (rank == 0) {
        printf("Total parallelism: %d threads\n", size * threadsPerRank);
    }

    GDALAllRegister();

    // All ranks read the data (OSM is read-only for searches)
    if (rank == 0) printf("\nReading TDA features (Speed Limits)...\n");
    std::vector<OGRFeature*> floridaTDA = ReadFeatures(
        PATH_TO_FLORIDA_MAX_SPEED_LIMIT, "Maximum_Speed_Limit_TDA");

    if (rank == 0) printf("\nReading OSM data (Roads, filtered)...\n");
    std::vector<OGRFeature*> floridaOSM = ReadFeatures(
        PATH_TO_FLORIDA_OSM, "lines", "highway IS NOT NULL");

    // Wait for all ranks to finish loading
    MPI_Barrier(MPI_COMM_WORLD);

    // Only rank 0 prints sample features
    if (rank == 0) {
        printf("\nFirst 5 TDA features:\n");
        for (int i = 0; i < 5 && i < floridaTDA.size(); ++i)
            PrintFeature(floridaTDA[i]);

        printf("\nFirst 5 OSM features:\n");
        for (int i = 0; i < 5 && i < floridaOSM.size(); ++i)
            PrintFeature(floridaOSM[i]);
    }

    int match_limit = 100;

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // Perform hybrid parallel matching
    std::vector<Match> localMatches = HybridGreedyMatch(
        floridaTDA, floridaOSM, match_limit, rank, size);

    if (rank == 0) printf("\nGathering results from all ranks...\n");

    // Gather results to rank 0
    std::vector<Match> allMatches = GatherMatches(
        localMatches, floridaTDA, floridaOSM, match_limit, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    if (rank == 0) {
        PrintMatches(allMatches);
    }

    printf("Matching took %f seconds\n", endTime - startTime);

    // Cleanup
    for (auto f : floridaTDA) OGRFeature::DestroyFeature(f);
    for (auto f : floridaOSM) OGRFeature::DestroyFeature(f);

    MPI_Finalize();
    return 0;
}