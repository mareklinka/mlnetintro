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

            var context = new MLContext();

            // this is necessary for calling Fit() later
            var fakeData = context.Data.ReadFromEnumerable(new List<InputModel>());

            // loads images from disk
            var imageLoadingEstimator = new ImageLoadingEstimator(
                context, 
                imagePath, 
                ("ImageData", nameof(InputModel.Path)));

            // resizes images to required size
            var imageResizingEstimator = new ImageResizingEstimator(
                context,
                "ImageResized",
                64,
                64,
                "ImageData",
                ImageResizerTransformer.ResizingKind.IsoPad);

            // transforms images to float vectors
            var imagePixelExtractingEstimator = new ImagePixelExtractingEstimator(context,
                new ImagePixelExtractorTransformer.ColumnInfo(
                    "input_1",
                    "ImageResized",
                    ImagePixelExtractorTransformer.ColorBits.Rgb, interleave: true,
                    scale: 1/255f));

            // loads the TF model from disk
            var tensorFlowEstimator = new TensorFlowEstimator(
                context, 
                new[] {"final_layer/Sigmoid"},
                new[] {"input_1"},
                "Model/MalariaModel.pb");

            // create a ML pipeline
            var pipeline = imageLoadingEstimator
                .Append(imageResizingEstimator)
                .Append(imagePixelExtractingEstimator)
                .Append(tensorFlowEstimator);

            // fit, otherwise we can't create a prediction engine
            var model = pipeline.Fit(fakeData);

            var predictor = model.CreatePredictionEngine<InputModel, OutputModel>(context);

            foreach (var file in Directory.GetFiles(imagePath, "*.png").Take(100).ToList())
            {
                var newName = ImageUtilities.ResizeImage(file, 64, 64, imagePath);

                var prediction = predictor.Predict(new InputModel {Path = newName});

                Console.WriteLine($"{Path.GetFileName(file)}: {(prediction.IsInfected() ? "Infected" : "Healthy")}");
            }

            Console.ReadLine();
        }
    }
}
