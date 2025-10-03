namespace Neiron;

public class Neiron
{
    public double[] Weights;
    public double Bias;
    private static Random rand = new();

    public Neiron(int inputCount)
    {
        Weights = new double[inputCount];
        for (int i = 0; i < inputCount; i++)
            Weights[i] = rand.NextDouble() * 2 - 1; // [-1, 1]

        Bias = rand.NextDouble() * 2 - 1;
    }

    public double FeedForward(double[] inputs)
    {
        double sum = 0.0;
        for (int i = 0; i < inputs.Length; i++)
            sum += inputs[i] * Weights[i];

        sum += Bias;
        return Sigmoid(sum);
    }

    public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public static double SigmoidDerivative(double x)
    {
        double s = Sigmoid(x);
        return s * (1 - s);
    }
}
