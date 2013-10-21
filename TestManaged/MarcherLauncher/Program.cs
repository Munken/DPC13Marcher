using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MarcherLauncher
{
    using ManagedCuda;
    using ManagedCuda.VectorTypes;
    class Program
    {
        static void Main(string[] args)
        {
            CudaContext cntxt = new CudaContext();
            var cumodule = cntxt.LoadModule(@"..\..\..\Marcher\Debug\march3.ptx");
            var kernel = new CudaKernel("simpleKernel", cumodule, cntxt);
            kernel.SetConstantVariable("d_edgeTable", Tables.EDGE_TABLE);
            kernel.SetConstantVariable("d_triTable", Tables.TRI_TABLE);

            dim3 d = new dim3(10, 10, 10);
            uint N = prod(d);
            float iso = 0;
            float3 min = new float3(0, 0, 0);
            float3 stepsize = new float3(0.2f, 0.2f, 0.2f);
            
            CudaDeviceVariable<float3> result = new CudaDeviceVariable<float3>(N);
            CudaDeviceVariable<uint> count = new CudaDeviceVariable<uint>(N);

            kernel.Run(iso, d, min, stepsize, result.DevicePointer, count.DevicePointer);
            

            // (float isoValue, dim3 dims, float3 minX, float3 dx, float3* out, uint* count)
        }

        static uint prod(dim3 d)
        {
            return d.x * d.y * d.z;
        }
    }
}
