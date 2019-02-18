using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.ImageAnalytics;
using Microsoft.ML.Transforms;

namespace BA.NET.MLNET.Demo2
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Write("Image directory: ");
            var imagePath = Console.ReadLine();

            var env = new MLContext(conc: 0);

            var data = env.Data.ReadFromEnumerable(new List<InputModel>());

            var imageLoadingEstimator = new ImageLoadingEstimator(env, imagePath, ("ImageData", nameof(InputModel.Path)));

            var imageResizingEstimator = new ImageResizingEstimator(env,
                "ImageResized",
                64,
                64,
                "ImageData",
                ImageResizerTransformer.ResizingKind.IsoPad);

            var imagePixelExtractingEstimator = new ImagePixelExtractingEstimator(env,
                new ImagePixelExtractorTransformer.ColumnInfo(
                    "input_1",
                    "ImageResized",
                    ImagePixelExtractorTransformer.ColorBits.Rgb, interleave: true,
                    scale: 1/255f));

            var tensorFlowEstimator = new TensorFlowEstimator(env, new[] {"final_layer/Sigmoid"},
                new[] {"input_1"}, "Model/MalariaModel.pb");

            var pipeline = imageLoadingEstimator
                .Append(imageResizingEstimator)
                .Append(imagePixelExtractingEstimator)
                .Append(tensorFlowEstimator);

            var model = pipeline.Fit(data);
            var predictor = model.CreatePredictionEngine<InputModel, OutputModel>(env);

            foreach (var file in Directory.GetFiles(imagePath, "*.png").Take(100).ToList())
            {
                var (newName, _, _) = ImageUtilities.FlipRotateImage(file, 64, 64, imagePath);

                var prediction = predictor.Predict(new InputModel {Path = newName});

                Console.WriteLine($"{Path.GetFileName(file)}: {(prediction.IsInfected() ? "Infected" : "Healthy")}");
            }

            Console.ReadLine();
        }
    }
}
