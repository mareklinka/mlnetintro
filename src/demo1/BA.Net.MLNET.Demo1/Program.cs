using System;
using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace BA.Net.MLNET.Demo1
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var trainingData =
                context.Data.ReadFromTextFile<TrainingInputModel>("data\\train.csv", hasHeader: true, separatorChar: ',');

            var pipeline = context.Transforms.DropColumns(nameof(TrainingInputModel.PassengerId), nameof(TrainingInputModel.Name),
                    nameof(TrainingInputModel.Ticket), nameof(TrainingInputModel.Fare), nameof(TrainingInputModel.Cabin))
                .Append(context.Transforms.ReplaceMissingValues(nameof(TrainingInputModel.Age), nameof(TrainingInputModel.Age),
                    MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean))
                .Append(context.Transforms.Categorical.OneHotEncoding(nameof(TrainingInputModel.Gender),
                    nameof(TrainingInputModel.Gender)))
                .Append(context.Transforms.Categorical.OneHotEncoding(nameof(TrainingInputModel.Embarked),
                    nameof(TrainingInputModel.Embarked)))
                .Append(context.Transforms.Categorical.OneHotEncoding(nameof(TrainingInputModel.PassengerClass),
                    nameof(TrainingInputModel.PassengerClass)))
                .Append(context.Transforms.Concatenate("Features", nameof(TrainingInputModel.PassengerClass),
                    nameof(TrainingInputModel.Gender), nameof(TrainingInputModel.Age), nameof(TrainingInputModel.SiblingsOrSpouses),
                    nameof(TrainingInputModel.ParentsOrChildren), nameof(TrainingInputModel.Embarked)))
                .Append(context.BinaryClassification.Trainers.FastTree(nameof(TrainingInputModel.Survived)))
                .Fit(trainingData);

            var evalData = context.Data.ReadFromTextFile<PredictionInputModel>("data\\test.csv", hasHeader: true, separatorChar: ',');
            var statistics = context.BinaryClassification.EvaluateNonCalibrated(pipeline.Transform(trainingData), nameof(OutputModel.Survived));

            Console.WriteLine($"Accuracy: {statistics.Accuracy}");
            Console.WriteLine($"F1: {statistics.F1Score}");

            Console.WriteLine();

            var predictor = pipeline.CreatePredictionEngine<PredictionInputModel, OutputModel>(context);

            foreach (var row in evalData.Preview().RowView)
            {
                var predictionInputModel = new PredictionInputModel
                {
                    Embarked = (row.Values[10].Value.ToString()),
                    PassengerClass = ((float)row.Values[1].Value),
                    Gender = (row.Values[3].Value.ToString()),
                    Age = ((float)row.Values[4].Value),
                    ParentsOrChildren = ((float)row.Values[6].Value),
                    SiblingsOrSpouses = ((float)row.Values[5].Value),
                    Cabin = (row.Values[9].Value.ToString()),
                    Name = (row.Values[2].Value.ToString()),
                    Fare = ((double)row.Values[8].Value),
                    Ticket = (row.Values[7].Value.ToString()),
                    PassengerId = ((int) row.Values[0].Value)
                };
                var prediction = predictor.Predict(predictionInputModel);

                Console.WriteLine($"{predictionInputModel.Name}: {(prediction.Survived ? "Alive" : "Deceased")} ({prediction.Probability:P2})");
            }
        }
    }
}
