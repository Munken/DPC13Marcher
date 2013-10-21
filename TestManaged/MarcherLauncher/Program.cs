using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MarcherLauncher
{
    using ManagedCuda;
    class Program
    {
        static void Main(string[] args)
        {
            CudaContext cntxt = new CudaContext();
            var cumodule = cntxt.LoadModule(@"..\..\..\Marcher\Debug\march3.ptx");
            var kernel = new CudaKernel("simpleKernel", cumodule, cntxt);
            kernel.SetConstantVariable("d_edgeTable", Tables.EDGE_TABLE);
            kernel.SetConstantVariable("d_triTable", Tables.TRI_TABLE);

        }
    }
}
