using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using ManagedCuda;

namespace ManagedTest
{
    class Program
    {
        const int VECTOR_SIZE = 5120;
        const int THREADS_PER_BLOCK = 256;

        static CudaKernel fillVectorWithCuda;

        static void InitKernels()
        {
            CudaContext cntxt = new CudaContext();
            var cumodule = cntxt.LoadModule(@"..\..\..\TestManaged\Debug\kernel.ptx");
            fillVectorWithCuda = new CudaKernel("kernel", cumodule, cntxt);
            fillVectorWithCuda.BlockDimensions = THREADS_PER_BLOCK;
            fillVectorWithCuda.GridDimensions = VECTOR_SIZE / THREADS_PER_BLOCK + 1;
        }

        static Func<int[], int, int[]> fillVector = (m, value) =>
        {
            // init parameters
            CudaDeviceVariable<int> vector_host = m;
            // run cuda method
            fillVectorWithCuda.Run(vector_host.DevicePointer, value);
            // copy return to host
            int[] output = new int[m.Length];
            vector_host.CopyToHost(output);
            return output;
        };

        static void Main(string[] args)
        {
            InitKernels();
            int[] vector = Enumerable.Range(1, VECTOR_SIZE).ToArray();
            vector = fillVector(vector, 13);
            foreach (int s in vector) { Console.WriteLine(s); }
            Console.ReadKey();
        }
    }
}
