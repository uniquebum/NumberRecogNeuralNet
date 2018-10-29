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
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class NetworkTest {

    public int correctGuesses(int loops) {
        //HiddenLayer size
        int hLayerN1 = 10;
        int inputN = 784;
        int inputPartN = 196; // 784/4
        int outputN = 10;

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
            file = Paths.get("biases1_2.txt");
            saveBiases1 = Files.readAllLines(file,Charset.forName("UTF-8"));
            file = Paths.get("biases2_2.txt");
            saveBiases2 = Files.readAllLines(file,Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.out.println("WARNING: EXCEPTION OCCURRED");
        }

        //Init initialWeights
        //Init biases
        double[][][] weights1 = new double[4][hLayerN1][inputPartN]; //28^2
        double[][] weights2 = new double[outputN][hLayerN1*4];
        double[][] bias1 = new double[4][hLayerN1];
        double[] bias2 = new double[outputN];

        //Init weights
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < hLayerN1; k++) {
                for (int i = 0; i < 196; i++) {
                    weights1[j][k][i] = Double.parseDouble(saveWeights1.get(i + k*196 + j*hLayerN1*196));
                }
            }
        }
        for (int i = 0; i < outputN; i++) {
            for (int k = 0; k < hLayerN1*4; k++) {
                weights2[i][k] = Double.parseDouble(saveWeights2.get(k+i*hLayerN1*4));
            }
        }
        //Init biases
        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < hLayerN1; i++) {
                bias1[j][i] = Double.parseDouble(saveBiases1.get(i+j*hLayerN1));
            }
        }
        for (int i = 0; i < 10; i++) {
            bias2[i] = Double.parseDouble(saveBiases2.get(i));
        }

        double[] result = new double[outputN];
        int[] targetResult = new int[outputN];
        double[][] wSum1 = new double[4][hLayerN1];
        double[] wSum2 = new double[outputN];
        double[] hiddenNeuron1 = new double[hLayerN1*4];
        int guessProb = 0;
        int totalLoops = loops;
        int targetNumber;
        int imageNumber = 1;

        //Neural network loop
        while (imageNumber < totalLoops) {
            targetNumber = 0;
            while (targetNumber < 10) {
                //FORWARD PROPAGATION
                for (int i = 0; i < targetResult.length; i++) {
                    targetResult[i] = 0;
                }
                targetResult[targetNumber] = 1;

                double[][] inputs = calculateBlackWhiteValues(targetNumber, imageNumber); //Get color shades from random image
                inputs = normalizeInputs(inputs); //Normalize inputs to a range of 0 to 1

                //Calculate wsum for each hLayerN1 neurons
                for (int k = 0; k < 4; k++) {
                    for (int i = 0; i < hLayerN1; i++) {
                        wSum1[k][i] = weightedSum(inputs[k], weights1[k][i]) + bias1[k][i];
                        hiddenNeuron1[i + k * 10] = sigmoid(wSum1[k][i]);
                    }
                }

                //Calculate wsum for each hLayerN2 neurons
                for (int i = 0; i < outputN; i++) {
                    wSum2[i] = weightedSum(hiddenNeuron1, weights2[i]) + bias2[i];
                    result[i] = sigmoid(wSum2[i]);
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

    double[][] normalizeInputs(double[][] inputs) {
        double[][] y = new double[4][196];
        double[] inputs2 = new double[784];
        for (int i = 0; i < 4; i++) {
            for (int k = 0; k < 196; k++) {
                inputs2[k+i*196] = inputs[i][k];
            }
        }
        for (int k = 0; k < inputs.length; k++) {
            for (int i = 0; i < inputs[0].length; i++) {
                y[k][i] = (inputs[k][i] - calculateMean(inputs2)) / calculateVariance(inputs2);
            }
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

    public double[][] calculateBlackWhiteValues(int number, int imageNumber) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File("C:/Users/miror/IdeaProjects/MNIST_VectorToJPEG/src/com/company/" +
                    "MNIST/TrainImages/images" + Integer.toString(number) + "_0" + Integer.toString(imageNumber) + ".png"));
        } catch (IOException e) {

        }
        double[][] shades = new double[4][196];
        int avg;
        int A; int R; int G; int B;
        int ind0 = 0; int ind1 = 0; int ind2 = 0; int ind3 = 0;
        for (int i = 0; i < 28; i++) {
            for (int k = 0; k < 28; k++) {
                int rgbVal = img.getRGB(k, i);
                A = (rgbVal >> 24) & 0xff;
                R = (rgbVal >> 16) & 0xff;
                G = (rgbVal >> 8) & 0xff;
                B = rgbVal & 0xff;
                avg = (R + G + B) / 3;


                if (i < 14 && k < 14) {
                    shades[0][ind0] = avg; //A << 24 | avg << 16 | avg << 8 | avg;
                    ind0++;
                } else if (i >= 14 && k < 14) {
                    shades[1][ind1] = avg;
                    ind1++;
                } else if (i < 14 && k >= 14) {
                    shades[2][ind2] = avg;
                    ind2++;
                } else {
                    shades[3][ind3] = avg;
                    ind3++;
                }
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