package de.cogmod.rgnns;

import java.util.Random;

public class LearningESN {
    public static void main(String[] args) {

        final double[][] sequence = {{1,2},{1,2}};
        final int washout = 10;
        final int training = 10;
        final int test = 0;

        final EchoStateNetwork esn = new EchoStateNetwork(1, 3, 1);
        esn.initializeWeights(new Random(1234), 0.1);
        esn.trainESN(sequence,washout,training,test);

    }
}
