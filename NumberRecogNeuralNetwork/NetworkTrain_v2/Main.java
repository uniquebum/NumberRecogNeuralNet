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

public class Main {

    public double learnRate = 5;

    public static void main(String[] args) {
        new Main();
    }

    public Main() {
        //HiddenLayer size
        int hLayerN1 = 10;
        int inputN = 784;
        int inputPartN = 196; // 784/4
        int outputN = 10;

        //Init initialWeights
        double[][][] weights1 = new double[4][hLayerN1][inputPartN]; //28^2
        double[][] weights2 = new double[outputN][hLayerN1*4];

        //Randomize initial weights
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < hLayerN1; k++) {
                for (int i = 0; i < inputPartN; i++) {
                    weights1[j][k][i] = Math.random() - .5;
                }
                for (int i = 0; i < outputN; i++) {
                    weights2[i][k+10*j] = Math.random() - .5;
                }
            }
        }

        double[] result = new double[outputN];
        int[] targetResult = new int[outputN];
        double[][] wSum1 = new double[4][hLayerN1];
        double[] wSum2 = new double[outputN];
        double[] hiddenNeuron1 = new double[hLayerN1*4];
        double[] deltaHiddenNeuron1 = new double[hLayerN1*4];
        double[][] bias1 = new double[4][hLayerN1];
        double[][] deltaBias1 = new double[4][hLayerN1];
        double[] bias2 = new double[outputN];
        double[] deltaBias2 = new double[outputN];
        double[][][] deltaWeights1 = new double[4][hLayerN1][inputPartN];
        double[][] deltaWeights2 = new double[outputN][hLayerN1*4];
        double costFunction = 0;
        double costFunctionAv = 0;
        double costFunctionLimit = 0.25;
        double guessProb;
        double guessProbAv = 0;
        int iterations = 1;
        double backPropTerm;
        int loopCounter;
        int epochLength;
        int targetNumber = 0;

        //Randomize biases
        for (int k = 0; k < 4; k++) {
            for (int i = 0; i < hLayerN1; i++) {
                bias1[k][i] = Math.random() - .5;
            }
        }
        for (int i = 0; i < outputN; i++) {
            bias2[i] = Math.random() - .5;
        }

        //Get starting time
        DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
        Date dateobj = new Date();

        epochLength = 10;
        //Neural network loop
        while (true) {
            guessProb = 0;
            loopCounter = 0;
            costFunctionAv = 0;

            while (loopCounter < epochLength) {
                costFunction = 0;

                //FORWARD PROPAGATION
                for (int i = 0; i < targetResult.length; i++) {
                    targetResult[i] = 0;
                }
                targetNumber = loopCounter % 10;//randomIntWithRange(0, 9);
                int imageNumber = randomIntWithRange(1, 5421);
                targetResult[targetNumber] = 1;

                double[][] inputs = calculateBlackWhiteValues(targetNumber, imageNumber); //Get color shades from random image
                inputs = normalizeInputs(inputs); //Normalize inputs to a range of 0 to 1

                //Calculate wsum for each hLayerN1 neurons
                for (int k = 0; k < 4; k++) {
                    for (int i = 0; i < hLayerN1; i++) {
                        wSum1[k][i] = weightedSum(inputs[k], weights1[k][i]) + bias1[k][i];
                        hiddenNeuron1[i+k*10] = sigmoid(wSum1[k][i]);
                    }
                }

                //Calculate wsum for each hLayerN2 neurons
                for (int i = 0; i < outputN; i++) {
                    wSum2[i] = weightedSum(hiddenNeuron1, weights2[i]) + bias2[i];
                    result[i] = sigmoid(wSum2[i]);

                    costFunction += Math.pow(targetResult[i] - result[i], 2);// / (double)epochLength;
                }
                costFunctionAv += costFunction/(double)epochLength;


                //START BACKPROPAGATION
                for (int i = 0; i < 10; i++) {
                    backPropTerm = -sigmoidDerivative(wSum2[i])*2*(targetResult[i]-result[i]);
                    for (int k = 0; k < hLayerN1*4; k++) {
                        deltaWeights2[i][k] += hiddenNeuron1[k]*backPropTerm;///(double)epochLength;
                        deltaHiddenNeuron1[k] += weights2[i][k]*backPropTerm;
                    }
                    deltaBias2[i] += backPropTerm;///(double)epochLength;
                }

                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < hLayerN1; i++) {
                        backPropTerm = sigmoidDerivative(wSum1[j][i]) * deltaHiddenNeuron1[i+j*hLayerN1];
                        for (int k = 0; k < inputPartN; k++) {
                            deltaWeights1[j][i][k] += inputs[j][k] * backPropTerm;///(double)epochLength;
                        }
                        deltaBias1[j][i] += backPropTerm;///(double)epochLength;
                        deltaHiddenNeuron1[i+j*hLayerN1] = 0;
                    }
                }

                if (maxValInd(result) == targetNumber) {
                    guessProb += 1;
                }

                loopCounter++;
            }

            iterations++;

            if (iterations % 50 == 0) {
                System.out.println(Double.toString(guessProb) + " / " + Integer.toString(epochLength) + " --- " + Double.toString(costFunctionAv) + " " + Integer.toString(iterations));
            }
            if (iterations % 5000 == 0) {
                if (iterations == 5000) {
                    learnRate = 3;
                } else if (iterations == 10000) {
                    learnRate = 1;
                } /*else if (iterations == 15000) {
                    learnRate = .001;
                }*/
                System.out.println("Learn rate changed to: " + Double.toString(learnRate));
                List<String> saveWeights1 = new ArrayList<>();//Arrays.asList("The first line", "The second line");
                List<String> saveWeights2 = new ArrayList<>();
                List<String> saveBiases1 = new ArrayList<>();
                List<String> saveBiases2 = new ArrayList<>();
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < hLayerN1; i++) {
                        for (int k = 0; k < inputPartN; k++) {
                            saveWeights1.add(Double.toString(weights1[j][i][k]));
                        }
                    }
                }
                for (int i = 0; i < 10; i++) {
                    for (int k = 0; k < hLayerN1*4; k++) {
                        saveWeights2.add(Double.toString(weights2[i][k]));
                    }
                }
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < hLayerN1; i++) {
                        saveBiases1.add(Double.toString(bias1[j][i]));
                    }
                }
                for (int i = 0; i < 10; i++) {
                    saveBiases2.add(Double.toString(bias2[i]));
                }
                try {
                    Path file = Paths.get("weights1_2.txt");
                    Files.write(file, saveWeights1, Charset.forName("UTF-8"));
                    file = Paths.get("weights2_2.txt");
                    Files.write(file, saveWeights2, Charset.forName("UTF-8"));
                    file = Paths.get("biases1_2.txt");
                    Files.write(file, saveBiases1, Charset.forName("UTF-8"));
                    file = Paths.get("biases2_2.txt");
                    Files.write(file, saveBiases2, Charset.forName("UTF-8"));
                    //Files.write(file, saveWeights2, Charset.forName("UTF-8"), StandardOpenOption.APPEND);
                } catch (IOException e) {
                    System.out.println("WARNING: EXCEPTION OCCURRED!");
                }

                NetworkTest nwTest = new NetworkTest();
                int correctGuesses = nwTest.correctGuesses(250);
                System.out.println("Correct guesses: " + Integer.toString(correctGuesses) + " / 2500");

                if (correctGuesses > 250*9) {
                    System.out.println("Final results: \n" + Double.toString(guessProb) + " " +
                            Double.toString(costFunctionAv) + " " + Integer.toString(iterations));
                    System.out.println(Integer.toString(targetNumber) + " " + Double.toString(result[0]) + " " + Double.toString(result[1]) + " " +
                            Double.toString(result[2]) + " " + Double.toString(result[3]) + " " +
                            Double.toString(result[4]) + " " + Double.toString(result[5]) + " " +
                            Double.toString(result[6]) + " " + Double.toString(result[7]) + " " +
                            Double.toString(result[8]) + " " + Double.toString(result[9]));
                    System.out.println("Start time: " + df.format(dateobj));
                    dateobj = new Date();
                    System.out.println("End time: " + df.format(dateobj));
                    return;
                }
                costFunctionLimit = costFunctionLimit*0.9;
            }

            for (int i = 0; i < outputN; i++) {
                for (int k = 0; k < hLayerN1*4; k++) {
                    weights2[i][k] += -(deltaWeights2[i][k])*learnRate/(double)epochLength;
                    deltaWeights2[i][k] = 0;
                }
                bias2[i] += -deltaBias2[i]*learnRate/(double)epochLength;
                deltaBias2[i] = 0;
            }

            for (int j = 0; j < 4; j++) {
                for (int i = 0; i < hLayerN1; i++) {
                    for (int k = 0; k < inputPartN; k++) {
                        weights1[j][i][k] += -(deltaWeights1[j][i][k]) * learnRate / (double) epochLength;
                        deltaWeights1[j][i][k] = 0;
                    }
                    bias1[j][i] += -deltaBias1[j][i] * learnRate / (double) epochLength;
                    deltaBias1[j][i] = 0;
                }
            }
        }
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