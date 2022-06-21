package de.cogmod.rgnns;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class LearningESN {
    public static void main(String[] args) {

        final double[][] sequence = {{1, 2}, {1, 2}};
        final int washout = 10;
        final int training = 10;
        final int test = 0;

        final EchoStateNetwork esn = new EchoStateNetwork(1, 3, 1);
        esn.initializeWeights(new Random(1234), 0.1);
        //esn.trainESN(sequence,washout,training,test);

        try {
            double[][] Seq = loadSequence("data/sequence.txt");

            System.out.println(Seq.length);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
        public static double[][] loadSequence(final String filename) throws FileNotFoundException, IOException {
            return loadSequence(new FileInputStream(filename));
        }

        //**
        //* Helper method for sequence loading from InputStream.
        //
        public static double[][] loadSequence(final InputStream inputstream) throws IOException {
            //
            final BufferedReader input = new BufferedReader(
                    new InputStreamReader(inputstream));
            //
            final List<String[]> data = new ArrayList<String[]>();
            int maxcols = 0;
            //
            boolean read = true;
            //
            while (read) {
                final String line = input.readLine();

                if (line != null) {
                    final String[] components = line.trim().split("\\s*(,|\\s)\\s*");
                    final int cols = components.length;
                    if (cols > maxcols) {
                        maxcols = cols;
                    }
                    data.add(components);
                } else {
                    read = false;
                }
            }
            input.close();
            //
            final int cols = maxcols;
            final int rows = data.size();
            //
            if ((cols == 0) || (rows == 0)) return null;
            //
            final double[][] result = new double[rows][cols];
            //
            for (int r = 0; r < rows; r++) {
                String[] elements = data.get(r);
                for (int c = 0; c < cols; c++) {
                    final double value = Double.parseDouble(elements[c]);
                    result[r][c] = value;
                }
            }
            //
            return result;
        }


}
