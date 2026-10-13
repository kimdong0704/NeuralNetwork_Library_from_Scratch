import java.util.Random;

public class Lenet5Testing {

    public static void testDeterminism() {
        System.out.println("=== Determinism Test ===");
        System.out.println("Testing that same input produces same output...\n");

        // Create two networks with same random seed
        Random rand1 = new Random(12345);
        Random rand2 = new Random(12345);

        // Create fixed input
        double[][] input = new double[32][32];
        Random inputRand = new Random(67890);
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                input[i][j] = inputRand.nextDouble();
            }
        }

        // Note: We would need to modify lenet5 constructor to accept Random seed
        // For now, let's test that running the same network twice gives same results
        lenet5 net = new lenet5();

        double[] output1 = net.forward(input);
        double[] output2 = net.forward(input);

        System.out.println("First run output:");
        printOutput(output1);

        System.out.println("\nSecond run output:");
        printOutput(output2);

        // Check if outputs are identical
        boolean identical = true;
        for (int i = 0; i < output1.length; i++) {
            if (Math.abs(output1[i] - output2[i]) > 1e-10) {
                identical = false;
                break;
            }
        }

        System.out.println("\nOutputs identical: " + identical);
        if (identical) {
            System.out.println("✓ PASS: Network is deterministic");
        } else {
            System.out.println("✗ FAIL: Network is not deterministic");
        }
    }

    public static void testDifferentInputs() {
        System.out.println("\n=== Different Inputs Test ===");
        System.out.println("Testing that different inputs produce different outputs...\n");

        lenet5 net = new lenet5();

        // Create two different inputs
        double[][] input1 = new double[32][32];
        double[][] input2 = new double[32][32];

        Random rand1 = new Random(111);
        Random rand2 = new Random(222);

        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                input1[i][j] = rand1.nextDouble();
                input2[i][j] = rand2.nextDouble();
            }
        }

        double[] output1 = net.forward(input1);
        double[] output2 = net.forward(input2);

        System.out.println("Input 1 output:");
        printOutput(output1);

        System.out.println("\nInput 2 output:");
        printOutput(output2);

        // Check if outputs are different
        boolean different = false;
        for (int i = 0; i < output1.length; i++) {
            if (Math.abs(output1[i] - output2[i]) > 1e-6) {
                different = true;
                break;
            }
        }

        System.out.println("\nOutputs different: " + different);
        if (different) {
            System.out.println("✓ PASS: Network discriminates between inputs");
        } else {
            System.out.println("✗ FAIL: Network produces same output for different inputs");
        }
    }

    public static void testZeroInput() {
        System.out.println("\n=== Zero Input Test ===");
        System.out.println("Testing network with all-zero input...\n");

        lenet5 net = new lenet5();

        double[][] zeroInput = new double[32][32];
        // All zeros by default

        double[] output = net.forward(zeroInput);

        System.out.println("Zero input output:");
        printOutput(output);

        // Verify output is valid (sum to 1.0, all non-negative)
        double sum = 0.0;
        boolean allNonNegative = true;
        for (double prob : output) {
            sum += prob;
            if (prob < 0) allNonNegative = false;
        }

        System.out.printf("\nSum: %.6f\n", sum);
        System.out.println("All non-negative: " + allNonNegative);

        boolean valid = Math.abs(sum - 1.0) < 1e-6 && allNonNegative;
        if (valid) {
            System.out.println("✓ PASS: Zero input produces valid output");
        } else {
            System.out.println("✗ FAIL: Zero input produces invalid output");
        }
    }

    public static void testOnesInput() {
        System.out.println("\n=== Ones Input Test ===");
        System.out.println("Testing network with all-ones input...\n");

        lenet5 net = new lenet5();

        double[][] onesInput = new double[32][32];
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                onesInput[i][j] = 1.0;
            }
        }

        double[] output = net.forward(onesInput);

        System.out.println("Ones input output:");
        printOutput(output);

        // Verify output is valid
        double sum = 0.0;
        boolean allNonNegative = true;
        for (double prob : output) {
            sum += prob;
            if (prob < 0) allNonNegative = false;
        }

        System.out.printf("\nSum: %.6f\n", sum);
        System.out.println("All non-negative: " + allNonNegative);

        boolean valid = Math.abs(sum - 1.0) < 1e-6 && allNonNegative;
        if (valid) {
            System.out.println("✓ PASS: Ones input produces valid output");
        } else {
            System.out.println("✗ FAIL: Ones input produces invalid output");
        }
    }

    private static void printOutput(double[] output) {
        for (int i = 0; i < output.length; i++) {
            System.out.printf("  Digit %d: %.6f\n", i, output[i]);
        }
    }

    public static void main(String[] args) {
        System.out.println("LeNet-5 Testing Suite\n");

        testDeterminism();
        testDifferentInputs();
        testZeroInput();
        testOnesInput();

        System.out.println("\n=== Testing Complete ===");
    }
}
