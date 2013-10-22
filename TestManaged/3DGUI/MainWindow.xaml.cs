using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Media.Media3D;
using System.Diagnostics;
using Launcher;
using ManagedCuda.VectorTypes;
using ManagedCuda.BasicTypes;

namespace _3DGUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void simpleButton_Click(object sender, RoutedEventArgs e)
        {
            MeshGeometry3D triangleMesh = new MeshGeometry3D();
            Point3D point0 = new Point3D(0, 0, 0);
            Point3D point1 = new Point3D(5, 0, 0);
            Point3D point2 = new Point3D(0, 0, 5);

            triangleMesh.Positions.Add(point0);
            triangleMesh.Positions.Add(point1);
            triangleMesh.Positions.Add(point2);

            triangleMesh.TriangleIndices.Add(0);
            triangleMesh.TriangleIndices.Add(2);
            triangleMesh.TriangleIndices.Add(1);

            Vector3D normal = new Vector3D(0, 1, 0);
            triangleMesh.Normals.Add(normal);
            triangleMesh.Normals.Add(normal);
            triangleMesh.Normals.Add(normal);

            Material material = new DiffuseMaterial(new SolidColorBrush(Colors.DarkKhaki));
            GeometryModel3D triangleModel = new GeometryModel3D(triangleMesh, material);
            ModelVisual3D model = new ModelVisual3D();

            model.Content = triangleModel;
            this.mainViewport.Children.Add(model);
        }

        private void cubeButton_Click(object sender, RoutedEventArgs e)
        {
            Model3DGroup cube = new Model3DGroup();
            Point3D p0 = new Point3D(0, 0, 0);
            Point3D p1 = new Point3D(5, 0, 0);
            Point3D p2 = new Point3D(5, 0, 5);
            Point3D p3 = new Point3D(0, 0, 5);
            Point3D p4 = new Point3D(0, 5, 0);
            Point3D p5 = new Point3D(5, 5, 0);
            Point3D p6 = new Point3D(5, 5, 5);
            Point3D p7 = new Point3D(0, 5, 5);

            MeshGeometry3D mesh = new MeshGeometry3D();
            //front side triangles
            addToMesh(mesh,p3, p2, p6);
            addToMesh(mesh,p3, p6, p7);
            //right side triangles
            addToMesh(mesh,p2, p1, p5);
            addToMesh(mesh,p2, p5, p6);
            //back side triangles
            addToMesh(mesh,p1, p0, p4);
            addToMesh(mesh,p1, p4, p5);
            //left side triangles
            addToMesh(mesh,p0, p3, p7);
            addToMesh(mesh,p0, p7, p4);
            //top side triangles
            addToMesh(mesh,p7, p6, p5);
            addToMesh(mesh,p7, p5, p4);
            //bottom side triangles
            addToMesh(mesh,p2, p3, p0);
            addToMesh(mesh,p2, p0, p1);

            Material material = new DiffuseMaterial(
                new SolidColorBrush(Colors.DarkKhaki));
            GeometryModel3D geometry = new GeometryModel3D(
                mesh, material);
            cube.Children.Add(geometry);

            ModelVisual3D model = new ModelVisual3D();
            model.Content = cube;
            this.mainViewport.Children.Add(model);
        }

        private Model3DGroup CreateTriangleModel(Point3D p0, Point3D p1, Point3D p2)
        {
            MeshGeometry3D mesh = new MeshGeometry3D();
            mesh.Positions.Add(p0);
            mesh.Positions.Add(p1);
            mesh.Positions.Add(p2);
            mesh.TriangleIndices.Add(0);
            mesh.TriangleIndices.Add(1);
            mesh.TriangleIndices.Add(2);
            Vector3D normal = CalculateNormal(p0, p1, p2);
            mesh.Normals.Add(normal);
            mesh.Normals.Add(normal);
            mesh.Normals.Add(normal);
            Material material = new DiffuseMaterial(
                new SolidColorBrush(Colors.DarkKhaki));
            GeometryModel3D model = new GeometryModel3D(
                mesh, material);
            Model3DGroup group = new Model3DGroup();
            group.Children.Add(model);
            return group;
        }

        private Point3D toPoint(ref float3 f)
        {
            return new Point3D(f.x, f.y, f.z);
        }

        private Vector3D CalculateNormal(Point3D p0, Point3D p1, Point3D p2)
        {
            Vector3D v0 = new Vector3D(
                p1.X - p0.X, p1.Y - p0.Y, p1.Z - p0.Z);
            Vector3D v1 = new Vector3D(
                p2.X - p1.X, p2.Y - p1.Y, p2.Z - p1.Z);
            return Vector3D.CrossProduct(v0, v1);
        }

        private void ClearViewport()
        {
            ModelVisual3D m;
            for (int i = mainViewport.Children.Count - 1; i >= 0; i--)
            {
                m = (ModelVisual3D)mainViewport.Children[i];
                if (m.Content is DirectionalLight == false)
                    mainViewport.Children.Remove(m);
            }
        }

        private void topography_Click(object sender, RoutedEventArgs e)
        {
            ClearViewport();
            Stopwatch stopwatch = new Stopwatch();

            stopwatch.Start();

            var launcher = new MarcherLauncher();
            uint n = 100;
            launcher.dimensions = new dim3(1, 1, 1) * n;
            launcher.isoValue = 0;
            launcher.minValue = new float3(-1.5f, -1.5f, -1.5f);
            launcher.stepSize = new float3(1, 1, 1) * 0.05f;
            
            launcher.march();
            stopwatch.Stop();
            Debug.WriteLine("Time elapsed: {0}",
        stopwatch.ElapsedMilliseconds);

            stopwatch.Restart();
            MeshGeometry3D mesh = createMesh(launcher);
            stopwatch.Stop();
            Debug.WriteLine("Time elapsed: {0}", stopwatch.ElapsedMilliseconds);

            Material material = new DiffuseMaterial(
                new SolidColorBrush(Colors.DarkKhaki));
            GeometryModel3D model = new GeometryModel3D(
                mesh, material);
            Model3DGroup group = new Model3DGroup();
            group.Children.Add(model);
            ModelVisual3D visual = new ModelVisual3D();
            visual.Content = group;
            this.mainViewport.Children.Add(visual);
        }

        private MeshGeometry3D createMesh(MarcherLauncher launcher)
        {
            MeshGeometry3D mesh = new MeshGeometry3D();
            int N = launcher.count.Count();
            uint cubes = 0, triangles = 0;

            for (int i = 0; i < N; i++)
            {
                uint c = launcher.count[i];
                if (c == 0) continue;

                cubes++;
                triangles += c / 3;
                int offset = 15 * i;
                for (int j = 0; j < c; j += 3)
                {
                    float3 f0 = launcher.triangles[offset + j];
                    float3 f1 = launcher.triangles[offset + j + 1];
                    float3 f2 = launcher.triangles[offset + j + 2];
                    addToMesh(mesh, toPoint(ref f0), toPoint(ref f1), toPoint(ref f2));
                }
            }

            Debug.WriteLine("Cubes {0}, Triangles {1}", cubes, triangles);
            return mesh;
        }

        private void addToMesh(MeshGeometry3D mesh, Point3D p0, Point3D p1, Point3D p2)
        {
            mesh.Positions.Add(p0);
            mesh.Positions.Add(p1);
            mesh.Positions.Add(p2);
            mesh.TriangleIndices.Add(0);
            mesh.TriangleIndices.Add(1);
            mesh.TriangleIndices.Add(2);
            Vector3D normal = CalculateNormal(p0, p1, p2);
            mesh.Normals.Add(normal);
            mesh.Normals.Add(normal);
            mesh.Normals.Add(normal);
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            ClearViewport();
        }
    }
}
