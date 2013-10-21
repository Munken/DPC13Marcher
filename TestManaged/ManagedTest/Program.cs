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
        static CudaKernel addWithCuda;

        static void InitKernels()
        {
            CudaContext cntxt = new CudaContext();
            var cumodule = cntxt.LoadModule(@"..\..\..\TestManaged\Debug\kernel.ptx");
            addWithCuda = new CudaKernel("kernel", cumodule, cntxt);
        }

        static Func<int, int, int> cudaAdd = (a, b) =>
        {
            // init output parameters
            CudaDeviceVariable<int> result_dev = 0;
            int result_host = 0;
            // run CUDA method
            addWithCuda.Run(a, b, result_dev.DevicePointer);
            // copy return to host
            result_dev.CopyToHost(ref result_host);
            return result_host;
        };

        static void Main(string[] args)
        {
            InitKernels();
            Console.WriteLine(cudaAdd(3, 10));
            Console.ReadKey();
        }
    }
}
