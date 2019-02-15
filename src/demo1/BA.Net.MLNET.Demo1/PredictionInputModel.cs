using Microsoft.ML.Data;

namespace BA.Net.MLNET.Demo1
{
    public class PredictionInputModel
    {
        [LoadColumn(0)]
        public int PassengerId { get; set; }

        [LoadColumn(1)]
        public float PassengerClass { get; set; }

        [LoadColumn(2)]
        public string Name { get; set; }

        [LoadColumn(3)]
        public string Gender { get; set; }

        [LoadColumn(4)]
        public float Age { get; set; }

        [LoadColumn(5)]
        public float SiblingsOrSpouses { get; set; }

        [LoadColumn(6)]
        public float ParentsOrChildren { get; set; }

        [LoadColumn(7)]
        public string Ticket { get; set; }

        [LoadColumn(8)]
        public double Fare { get; set; }

        [LoadColumn(9)]
        public string Cabin { get; set; }

        [LoadColumn(10)]
        public string Embarked { get; set; }
    }
}