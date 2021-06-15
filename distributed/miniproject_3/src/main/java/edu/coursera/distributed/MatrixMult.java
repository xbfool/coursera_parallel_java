package edu.coursera.distributed;

import edu.coursera.distributed.util.MPI;
import edu.coursera.distributed.util.MPI.MPIException;

/**
 * A wrapper class for a parallel, MPI-based matrix multiply implementation.
 */
public class MatrixMult {
    /**
     * A parallel implementation of matrix multiply using MPI to express SPMD
     * parallelism. In particular, this method should store the output of
     * multiplying the matrices a and b into the matrix c.
     * <p>
     * This method is called simultaneously by all MPI ranks in a running MPI
     * program. For simplicity MPI_Init has already been called, and
     * MPI_Finalize should not be called in parallelMatrixMultiply.
     * <p>
     * On entry to parallelMatrixMultiply, the following will be true of a, b,
     * and c:
     * <p>
     * 1) The matrix a will only be filled with the input values on MPI rank
     * zero. Matrix a on all other ranks will be empty (initialized to all
     * zeros).
     * 2) Likewise, the matrix b will only be filled with input values on MPI
     * rank zero. Matrix b on all other ranks will be empty (initialized to
     * all zeros).
     * 3) Matrix c will be initialized to all zeros on all ranks.
     * <p>
     * Upon returning from parallelMatrixMultiply, the following must be true:
     * <p>
     * 1) On rank zero, matrix c must be filled with the final output of the
     * full matrix multiplication. The contents of matrix c on all other
     * ranks are ignored.
     * <p>
     * Therefore, it is the responsibility of this method to distribute the
     * input data in a and b across all MPI ranks for maximal parallelism,
     * perform the matrix multiply in parallel, and finally collect the output
     * data in c from all ranks back to the zeroth rank. You may use any of the
     * MPI APIs provided in the mpi object to accomplish this.
     * <p>
     * A reference sequential implementation is provided below, demonstrating
     * the use of the Matrix class's APIs.
     *
     * @param a   Input matrix
     * @param b   Input matrix
     * @param c   Output matrix
     * @param mpi MPI object supporting MPI APIs
     * @throws MPIException On MPI error. It is not expected that your
     *                      implementation should throw any MPI errors during
     *                      normal operation.
     */
    public static void parallelMatrixMultiply(Matrix a, Matrix b, Matrix c,
                                              final MPI mpi) throws MPIException {
        //mpi.MPI_Init();
        int rank = mpi.MPI_Comm_rank(mpi.MPI_COMM_WORLD);
        int num_procs = mpi.MPI_Comm_size(mpi.MPI_COMM_WORLD);
        if (rank == 0) {
            double[] buf_a = a.getValues();
            int size_a = a.getNCols() * a.getNRows();
            for (int other_rank = 1; other_rank < num_procs; other_rank++) {
                mpi.MPI_Send(buf_a,
                        0,
                        size_a,
                        other_rank,
                        0,
                        mpi.MPI_COMM_WORLD);
            }

            double[] buf_b = b.getValues();
            int size_b = b.getNCols() * b.getNRows();
            for (int other_rank = 1; other_rank < num_procs; other_rank++) {
                mpi.MPI_Send(buf_b,
                        0,
                        size_b,
                        other_rank,
                        1,
                        mpi.MPI_COMM_WORLD);
            }
        } else {
            int size_a = a.getNCols() * a.getNRows();
            double[] buf_a = new double[size_a];
            int size_b = b.getNCols() * b.getNRows();
            double[] buf_b = new double[size_b];
            mpi.MPI_Recv(buf_a, 0, size_a, 0, 0, mpi.MPI_COMM_WORLD);
            mpi.MPI_Recv(buf_b, 0, size_b, 0, 1, mpi.MPI_COMM_WORLD);
        }
        if (rank != 0) {
            for (int i = rank - 1; i < c.getNRows(); i += num_procs - 1) {
                for (int j = 0; j < c.getNCols(); j++) {
                    c.set(i, j, 0.0);
                    for (int k = 0; k < b.getNRows(); k++) {
                        c.incr(i, j, a.get(i, k) * b.get(k, j));
                    }
                }
                double[] buf_c = c.getValues();
                int size_c = c.getNCols();
                int offset = i * c.getNCols();
                mpi.MPI_Send(buf_c,
                        offset,
                        size_c,
                        0,
                        2,
                        mpi.MPI_COMM_WORLD);

            }
        } else {
            for (int i = 0; i < c.getNRows(); i++) {
                int size_c = c.getNCols();
                int offset = i * c.getNCols();
                double[] buf_c = new double[size_c];
                mpi.MPI_Recv(buf_c, offset, size_c, 0, 2, mpi.MPI_COMM_WORLD);
                for (int j = 0; j < c.getNCols(); j++) {
                    c.set(i, j, buf_c[j]);
                }
            }
        }
        //mpi.MPI_Finalize();
    }
}
