package com.company;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NetworkTest {

    public static void main(String[] args) {
        new NetworkTest();
    }

    public NetworkTest() {

    }

    public int correctGuesses(int loops) {
        //HiddenLayer size
        int hLayerN1 = 40;
        int hLayerN2 = 20;

        //Load weights and biases
        //Load weights and biases
        List<String> saveWeights1 = new ArrayList<>();//Arrays.asList("The first line", "The second line");
        List<String> saveWeights2 = new ArrayList<>();
        List<String> saveWeights3 = new ArrayList<>();
        List<String> saveBiases1 = new ArrayList<>();
        List<String> saveBiases2 = new ArrayList<>();
        List<String> saveBiases3 = new ArrayList<>();
        try {
            Path file = Paths.get("weights1_2.txt");
            saveWeights1 = Files.readAllLines(file,Charset.forName("UTF-8"));
            file = Paths.get("weights2_2.txt");
            saveWeights2 = Files.readAllLines(file,Charset.forName("UTF-8"));
            file = Paths.get("weights3_2.txt");
            saveWeights3 = Files.readAllLines(file,Charset.forName("UTF-8"));
            file = Paths.get("biases1_2.txt");
            saveBiases1 = Files.readAllLines(file,Charset.forName("UTF-8"));
            file = Paths.get("biases2_2.txt");
            saveBiases2 = Files.readAllLines(file,Charset.forName("UTF-8"));
            file = Paths.get("biases3_2.txt");
            saveBiases3 = Files.readAllLines(file,Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.out.println("WARNING: EXCEPTION OCCURRED");
        }

        //Init initialWeights
        double[][] weights1 = new double[hLayerN1][784]; //28^2
        double[][] weights2 = new double[hLayerN2][hLayerN1];
        double[][] weights3 = new double[10][hLayerN2];

        //Init weights
        for (int k = 0; k < hLayerN1; k++) {
            for (int i = 0; i < 784; i++) {
                weights1[k][i] = Double.parseDouble(saveWeights1.get(i+k*784));
            }
        }
        for (int i = 0; i < hLayerN2; i++) {
            for (int k = 0; k < hLayerN1; k++) {
                weights2[i][k] = Double.parseDouble(saveWeights2.get(k+i*hLayerN1));
            }
        }
        for (int k = 0; k < 10; k++) {
            for (int i = 0; i < hLayerN2; i++) {
                weights3[k][i] = Double.parseDouble(saveWeights3.get(i+k*hLayerN2));
            }
        }

        double[] result = new double[10];
        int[] targetResult = new int[10];
        double[] wSum1 = new double[hLayerN1];
        double[] wSum2 = new double[hLayerN2];
        double[] wSum3 = new double[10];
        double[] hiddenNeuron1 = new double[hLayerN1];
        double[] hiddenNeuron2 = new double[hLayerN2];
        double[] bias1 = new double[hLayerN1];
        double[] bias2 = new double[hLayerN2];
        double[] bias3 = new double[10];
        int guessProb = 0;
        int loopCounter = 0;
        int totalLoops = loops;
        int targetNumber;
        int imageNumber = 1;

        //Init biases
        for (int i = 0; i < hLayerN1; i++) {
            bias1[i] = Double.parseDouble(saveBiases1.get(i));
        }
        for (int i = 0; i < hLayerN2; i++) {
            bias2[i] = Double.parseDouble(saveBiases2.get(i));
        }
        for (int i = 0; i < 10; i++) {
            bias3[i] = Double.parseDouble(saveBiases3.get(i));
        }

        //Neural network loop
        while (imageNumber < totalLoops) {
            targetNumber = 0;
            while (targetNumber < 10) {
                //FORWARD PROPAGATION
                for (int i = 0; i < targetResult.length; i++) {
                    targetResult[i] = 0;
                }
                targetResult[targetNumber] = 1;

                double[] inputs = calculateBlackWhiteValues(targetNumber, imageNumber); //Get color shades from random image
                inputs = normalizeInputs(inputs); //Normalize inputs to a range of 0 to 1

                //Calculate wsum for each hLayerN neurons
                for (int i = 0; i < hLayerN1; i++) {
                    wSum1[i] = weightedSum(inputs, weights1[i]) + bias1[i];
                    hiddenNeuron1[i] = sigmoid(wSum1[i]);
                }

                //Calculate wsum from the hidden layer to get the final result and then calculate error
                for (int i = 0; i < hLayerN2; i++) {
                    wSum2[i] = weightedSum(hiddenNeuron1, weights2[i]) + bias2[i];
                    hiddenNeuron2[i] = sigmoid(wSum2[i]);
                }

                for (int i = 0; i < 10; i++) {
                    wSum3[i] = weightedSum(hiddenNeuron2, weights3[i]) + bias3[i];
                    result[i] = sigmoid(wSum3[i]);
                }

                if (maxValInd(result) == targetNumber) {
                    guessProb += 1;
                }

                targetNumber++;
            }
            imageNumber++;
        }

        return guessProb;
    }

    double[] normalizeInputs(double[] inputs) {
        double[] y = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            y[i] = (inputs[i]-calculateMean(inputs))/calculateVariance(inputs);
        }

        return y;
    }

    double calculateMean(double[] inputs) {
        double y = 0;
        for (int i = 0; i < inputs.length; i++) {
            y += inputs[i]/(double)inputs.length;
        }
        return y;
    }

    double calculateVariance(double[] inputs) {
        double y = 0;
        double meanVal = calculateMean(inputs);
        for (int i = 0; i < inputs.length; i++) {
            y += Math.pow(inputs[i]-meanVal,2)/inputs.length;
        }
        return y;
    }

    int randomIntWithRange(int min, int max)
    {
        int range = (max - min) + 1;
        return (int)(Math.random() * range) + min;
    }

    public double[] calculateBlackWhiteValues(int number, int imageNumber) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File("C:/Users/miror/IdeaProjects/MNIST_VectorToJPEG/src/com/company/" +
                    "MNIST/TestImages/images" + Integer.toString(number) + "_0" + Integer.toString(imageNumber) + ".png"));
        } catch (IOException e) {

        }
        double[] shades = new double[784];
        int avg;
        int A; int R; int G; int B;
        for (int i = 0; i < 28; i++) {
            for (int k = 0; k < 28; k++) {
                int rgbVal = img.getRGB(i, k);
                A = (rgbVal >> 24) & 0xff;
                R = (rgbVal >> 16) & 0xff;
                G = (rgbVal >> 8) & 0xff;
                B = rgbVal & 0xff;
                avg = (R + G + B) / 3;

                shades[i*28+k] = avg; //A << 24 | avg << 16 | avg << 8 | avg;
            }

        }

        return shades;
    }

    public double weightedSum(double[] activations, double[] weight) {
        double weightedSum = 0;
        for (int i = 0; i < activations.length; i++) {
            weightedSum += weight[i]*activations[i];
        }
        return weightedSum;
    }

    public double sigmoid(double x) {
        double y;
        y = 1/(1+Math.exp(-x));
        return y;
    }

    public double sigmoidDerivative(double x) {
        double y;
        y = sigmoid(x)*(1-sigmoid(x)); //1/(1+Math.exp(-x));
        return y;
    }

    public int maxValInd(double[] x) {
        int ind = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[ind] < x[i]) {
                ind = i;
            }
        }
        return ind;
    }


}
