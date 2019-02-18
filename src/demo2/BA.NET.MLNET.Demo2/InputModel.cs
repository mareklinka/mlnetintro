using Microsoft.ML.Data;

namespace BA.NET.MLNET.Demo2
{
    public class InputModel
    {
        public string Path { get; set; }
    }

    public class OutputModel
    {
        [ColumnName("final_layer/Sigmoid")]
        public float[] Prediction { get; set; }

        public bool IsInfected() => Prediction[0] < 0.5f;
    }
}
