using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Transforms;

namespace BA.Net.MLNET.Demo1
{
    class Program
    {
        static void Main(string[] args)
        {
            TrainModel();

            UseModel();

            Console.ReadLine();
        }

        private static void TrainModel()
        {
            var context = new MLContext();

            // load data from disk
            var trainingData =
                context.Data.ReadFromTextFile<TrainingInputModel>("data\\train.csv", hasHeader: true, separatorChar: ',');

            // create a learning pipeline:
            // drop unnecessary columns
            // deal with missing values
            // one-hot encode categorical columns
            // concatenate feature columns into a vector
            // add learner
            // learn
            var pipeline = context.Transforms.DropColumns(
                    nameof(TrainingInputModel.PassengerId),
                    nameof(TrainingInputModel.Name),
                    nameof(TrainingInputModel.Ticket), 
                    nameof(TrainingInputModel.Fare), 
                    nameof(TrainingInputModel.Cabin))
                .Append(context.Transforms.ReplaceMissingValues(
                    nameof(TrainingInputModel.Age), 
                    nameof(TrainingInputModel.Age),
                    MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean))
                .Append(context.Transforms.Categorical.OneHotEncoding(
                    nameof(TrainingInputModel.Gender),
                    nameof(TrainingInputModel.Gender)))
                .Append(context.Transforms.Categorical.OneHotEncoding(
                    nameof(TrainingInputModel.Embarked),
                    nameof(TrainingInputModel.Embarked)))
                .Append(context.Transforms.Categorical.OneHotEncoding(
                    nameof(TrainingInputModel.PassengerClass),
                    nameof(TrainingInputModel.PassengerClass)))
                .Append(context.Transforms.Concatenate("Features", 
                    nameof(TrainingInputModel.PassengerClass),
                    nameof(TrainingInputModel.Gender), 
                    nameof(TrainingInputModel.Age),
                    nameof(TrainingInputModel.SiblingsOrSpouses),
                    nameof(TrainingInputModel.ParentsOrChildren), 
                    nameof(TrainingInputModel.Embarked)))
                .Append(context.BinaryClassification.Trainers.LogisticRegression(nameof(TrainingInputModel.Survived)))
                .Fit(trainingData);

            EvaluateModel(context, pipeline);

            Console.WriteLine();

            PredictTestData(context, pipeline);

            Console.WriteLine();

            SaveModel(context, pipeline);
        }

        private static void EvaluateModel(MLContext context, ITransformer pipeline)
        {
            var trainingData =
                context.Data.ReadFromTextFile<TrainingInputModel>("data\\train.csv", hasHeader: true, separatorChar: ',');

            // evaluate on training data
            var statistics =
                context.BinaryClassification.EvaluateNonCalibrated(pipeline.Transform(trainingData),
                    nameof(OutputModel.Survived));

            Console.WriteLine("Training performance:");
            Console.WriteLine($"\tAccuracy: {statistics.Accuracy}");
            Console.WriteLine($"\tF1: {statistics.F1Score}");
        }

        private static void PredictTestData(MLContext context, ITransformer pipeline)
        {
            var predictor = pipeline.CreatePredictionEngine<PredictionInputModel, OutputModel>(context);

            var evalData =
                context.Data.ReadFromTextFile<PredictionInputModel>("data\\test.csv", hasHeader: true, separatorChar: ',');

            foreach (var row in evalData.Preview().RowView)
            {
                var inputModel = new PredictionInputModel
                {
                    Embarked = row.Values[10].Value.ToString(),
                    PassengerClass = (float)row.Values[1].Value,
                    Gender = row.Values[3].Value.ToString(),
                    Age = (float)row.Values[4].Value,
                    ParentsOrChildren = (float)row.Values[6].Value,
                    SiblingsOrSpouses = (float)row.Values[5].Value,
                    Cabin = row.Values[9].Value.ToString(),
                    Name = row.Values[2].Value.ToString(),
                    Fare = (double)row.Values[8].Value,
                    Ticket = row.Values[7].Value.ToString(),
                    PassengerId = (int)row.Values[0].Value
                };

                var prediction = predictor.Predict(inputModel);

                Console.WriteLine(
                    $"{inputModel.Name}: {(prediction.Survived ? "Alive" : "Deceased")} ({prediction.Probability:P2})");
            }
        }

        private static void SaveModel(MLContext context, ITransformer pipeline)
        {
            Console.WriteLine("Saving the model to model.bin...");
            using (var modelFileStream = new FileStream("model.bin", FileMode.Create, FileAccess.Write))
            {
                pipeline.SaveTo(context, modelFileStream);
            }
        }

        private static (MLContext, ITransformer) LoadModel()
        {
            Console.WriteLine("Loading the model from model.bin...");
            var context = new MLContext();

            using (var modelFileStream = new FileStream("model.bin", FileMode.Open, FileAccess.Read))
            {
                return (context, context.Model.Load(modelFileStream));
            }
        }

        private static void UseModel()
        {
            var (context, predictor) = LoadModel();

            Console.WriteLine();

            PredictTestData(context, predictor);
        }
    }
}
