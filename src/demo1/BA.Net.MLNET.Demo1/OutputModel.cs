using Microsoft.ML.Data;

namespace BA.Net.MLNET.Demo1
{
    public class OutputModel
    {
        [ColumnName("PredictedLabel")]
        public bool Survived;

        [ColumnName("Probability")]
        public float Probability;
    }
}