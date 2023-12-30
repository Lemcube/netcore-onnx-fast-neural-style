using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.Numerics.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;



namespace ConsoleApp3
{
    internal class Program
    {
        static void Main(string[] args)
        {
            const string mosaic = "./OnnxModels/mosaic-9.onnx";
            const string pointilism = "./OnnxModels/pointilism-8.onnx";
            const string udnie = "./OnnxModels/udnie-9.onnx";

            const string image1 = "lion.jpg";
            const string image2 = "red.png";

            string generationPrefix = DateTime.Now.ToString("yyMMdd_HHmmss_");
            ImgTransf(image1, mosaic, $"{generationPrefix}{nameof(mosaic)}.jpg");
            ImgTransf(image1, pointilism, $"{generationPrefix}{nameof(pointilism)}.jpg");
            ImgTransf(image1, udnie, $"{generationPrefix}{nameof(udnie)}.jpg");
        }


        public static void ImgTransf(string imagename, string modelPath, string destFileName)
        {
            // Read the JPEG image
            using Image<Rgb24> image = Image.Load<Rgb24>(imagename);

            if (image.Width != 224 || image.Height != 224)
            {
                // Resize the image to 224x224
                image.Mutate(x => x.Resize(224, 224));
            }

            // Convert the image to a tensor
            float[] tensorData = new float[1 * 3 * 224 * 224];
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    Rgb24 pixel = image[x, y];
                    tensorData[y * image.Width + x] = pixel.R;
                    tensorData[image.Width * image.Height + y * image.Width + x] = pixel.G;
                    tensorData[2 * image.Width * image.Height + y * image.Width + x] = pixel.B;
                }
            }

            var tensor = new DenseTensor<float>(tensorData, new[] { 1, 3, 224, 224 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input1", tensor)
            }; 

            using var session = new InferenceSession(modelPath);

            using var results = session.Run(inputs);

            foreach (var result in results)
            {
                Console.WriteLine($"Output: {result.Name}");
                var outputTensor = result.AsTensor<float>();


                var outputData = outputTensor.ToArray();

                //SELF TEST WITHOUT ML, just to be sure image decoding/encoding is simmetrical
                //var outputData = tensor.ToArray();

                using var imageOut = new Image<Rgb24>(224, 224);

                for (int y = 0; y < imageOut.Height; y++)
                {
                    for (int x = 0; x < imageOut.Width; x++)
                    {
                        // Calculate tensor index
                        int rIndex = y * image.Width + x;
                        int gIndex = image.Width * image.Height + y * image.Width + x;
                        int bIndex = 2 * image.Width * image.Height + y * image.Width + x;
                         
                        byte r = (byte)outputData[rIndex];
                        byte g = (byte)outputData[gIndex];
                        byte b = (byte)outputData[bIndex];
                         
                        imageOut[x, y] = new Rgb24(r, g, b);
                    }
                }

                // Save as JPEG
                imageOut.Save(destFileName);

            }

        }

         
        public void SimpleTest()
        {
            const string modelPath = "./single_add.onnx";
            //using var session = new InferenceSession(modelPath);

            using var session = new InferenceSession(modelPath);

            var inputDataA = new List<float> { 1.0f }; // Example value
            var inputDataB = new List<float> { 2.0f }; // Example value
            var inputDataE = new List<float> { 3.0f }; // Example value

            var tensorA = new DenseTensor<float>(inputDataA.ToArray(), new[] { 1 });
            var tensorB = new DenseTensor<float>(inputDataB.ToArray(), new[] { 1 });
            var tensorE = new DenseTensor<float>(inputDataE.ToArray(), new[] { 1 });

            // Create input container
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("A", tensorA),
                NamedOnnxValue.CreateFromTensor("B", tensorB),
                NamedOnnxValue.CreateFromTensor("E", tensorE)
            };

            // Run the model
            using var results = session.Run(inputs);

            // Extract and display results
            foreach (var result in results)
            {
                Console.WriteLine($"Output: {result.Name}");
                var tensor = result.AsTensor<float>();
                Console.WriteLine(tensor.GetValue(0));
            }
        }
    }
}
