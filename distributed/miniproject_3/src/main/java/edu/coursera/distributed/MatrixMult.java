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
        //    System.out.println("rank: " + rank + " total: " + num_procs);
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
                //           System.out.println("send a from: " + 0 + " to: " + other_rank);
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
                //           System.out.println("send b from: " + 0 + " to: " + other_rank);
            }
        } else {
            int size_a = a.getNCols() * a.getNRows();
            double[] buf_a = a.getValues();
            int size_b = b.getNCols() * b.getNRows();
            double[] buf_b = b.getValues();
            mpi.MPI_Recv(buf_a, 0, size_a, 0, 0, mpi.MPI_COMM_WORLD);
            //        System.out.println("recv a from: " + 0 + " to: " + rank);
            mpi.MPI_Recv(buf_b, 0, size_b, 0, 1, mpi.MPI_COMM_WORLD);
            //         System.out.println("recv b from: " + 0 + " to: " + rank);
        }

        int start = (c.getNRows() / num_procs) * rank;
        int end = (c.getNRows() / num_procs) * (rank + 1);
        int true_end = end > c.getNRows() ? c.getNRows() : end;
        for (int i = start; i < true_end; i++) {
            for (int j = 0; j < c.getNCols(); j++) {
                c.set(i, j, 0.0);
                for (int k = 0; k < b.getNRows(); k++) {
                    c.incr(i, j, a.get(i, k) * b.get(k, j));
                }
            }
        }
        //      System.out.println("rank: " + rank + "compute: " + "data from: " + start * c.getNCols() + " count: " + (true_end - start) * c.getNCols());
        if (rank != 0) {
            double[] buf_c = c.getValues();
            //           System.out.println("start send c from: " + rank + " to: " + 0 + "data from: " + start * c.getNCols() + " count: " + (true_end - start) * c.getNCols());
            int i_start = (c.getNRows() / num_procs) * rank;
            int i_end = (c.getNRows() / num_procs) * (rank + 1);
            int i_true_end = i_end > c.getNRows() ? c.getNRows() : i_end;
            int count = (i_true_end - i_start) * c.getNCols();
            mpi.MPI_Send(buf_c,
                    i_start * c.getNCols(),
                    count,
                    0,
                    2,
                    mpi.MPI_COMM_WORLD);
            //           System.out.println("end send c from: " + rank + " to: " + 0 + "data from: " + start * c.getNCols() + " count: " + (true_end - start) * c.getNCols());
        } else {
            for (int i = 1; i < num_procs; i++) {
                int i_start = (c.getNRows() / num_procs) * i;
                int i_end = (c.getNRows() / num_procs) * (i + 1);
                int i_true_end = i_end > c.getNRows() ? c.getNRows() : i_end;
                int count = (i_true_end - i_start) * c.getNCols();
                double[] buf_c = c.getValues();
                //System.out.println("start recv c from: " + i + " to: " + 0 + "data from: " + i_start * c.getNCols() + " count: " + count);
                mpi.MPI_Recv(buf_c, i_start * c.getNCols(), count, i, 2, mpi.MPI_COMM_WORLD);
                //System.out.println("end recv c from: " + i + " to: " + 0 + "data from: " + i_start * c.getNCols() + " count: " + count);
                //System.out.println("i_start " + i_start + " i_true_end " + i_true_end);
//                for (int j = i_start; j < i_true_end; j++) {
//                    for (int k = 0; k < c.getNCols(); k++) {
//                        c.set(j, k, buf_c[j * c.getNCols() + k]);
//                    }
//                }
//                System.out.println("rank " + i + "compute over");
            }
        }
//        System.out.println("rank: " + rank + " end");
    }
}
