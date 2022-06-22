package de.cogmod.rgnns;

import de.cogmod.rgnns.examples.ReservoirToolsExample;
import de.jannlab.optimization.BasicOptimizationListener;
import de.jannlab.optimization.Objective;
import de.jannlab.optimization.optimizer.DifferentialEvolution;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import de.jannlab.io.Serializer;

import static de.cogmod.rgnns.ReservoirTools.*;
import static de.cogmod.rgnns.ReservoirTools.matrixAsString;

public class DELearningESN {

    public static void main(String[] args) {

        double[][] Sequence = null;
        try {
            Sequence = loadSequence("data/Sequence.txt");  // pointer auf weights
        }
        catch (IOException e) {
            e.printStackTrace();
        }

        final int reservoirsize = 60;
        final double feedBackScaling = 1e-8;
        //final double[][] W_input = new double[3][reservoirsize];
        //final double[][] W_reservoir = new double[reservoirsize][reservoirsize];
        final EchoStateNetwork esn = new EchoStateNetwork(3, reservoirsize, 3);
        //esn.initializeWeights(new Random(1234), 0.1);
        int add_bias = 1;

        int washout = 100;
        int training = 1000;
        int test = 100;



        final Objective f = new Objective() {
            //
            @Override
            public int arity() {
                return (reservoirsize + add_bias)*(reservoirsize) + (3 +add_bias)*(reservoirsize); // Add Bias
            } // is W_reservoir + W_input
            @Override
            /**
             * This is the callback method that is called from the
             * optimizer to compute the "fitness" of a particular individual.
             */
            public double compute(double[] values, int offset) {

                // get an error if I don't include Sequence loading here, but should already be loaded in main
                double[][] Sequence = null;
                try {
                    Sequence = loadSequence("data/Sequence.txt");  // pointer auf weights
                }
                catch (IOException e) {
                    e.printStackTrace();
                }

                // Get the best Weights of the Population. Values stores the weights for the whole population,
                // the best starts at postion offset. There are arity many weights in every reservoir + Inputweights
                double[] optimizedWeights = Arrays.copyOfRange(values, offset, offset + arity());
                // The Inputweights get scaled by the OFB factor
                for (int i = 0; i < (reservoirsize ) * (3+ add_bias); i++) {
                    optimizedWeights[i] = optimizedWeights[i] * feedBackScaling;
                }

                // Read old Weights to copy the optimal SVD-calculated Outputweights from the training step
                double[] oldWeights = new double[esn.getWeightsNum()];
                double[] newWeights = new double[esn.getWeightsNum()];
                esn.readWeights(oldWeights);
                double[] Wout = Arrays.copyOfRange(oldWeights, arity(), esn.getWeightsNum());

                // Join the new optimized Reservoir + Inputweigths with the Outputweights and write to ESN
                // arraycopy: src array, src pos, dest arry, dest pos
                System.arraycopy(optimizedWeights, 0, newWeights, 0, optimizedWeights.length);
                System.arraycopy(Wout, 0, newWeights, optimizedWeights.length, Wout.length);
                esn.writeWeights(newWeights);  // flatten? Arrays.stream(W_reservoir).flatMapToInt(Arrays::stream).toArray();

                double error = esn.trainESN(Sequence, washout, training, test);
                return error;
            };
        };
        //
        // Now we setup the optimizer.
        //
        final DifferentialEvolution optimizer = new DifferentialEvolution();
        //
        // The same parameters can be used for reservoir optimization.
        //
        optimizer.setF(0.4);
        optimizer.setCR(0.6);
        optimizer.setPopulationSize(5);
        optimizer.setMutation(DifferentialEvolution.Mutation.CURR2RANDBEST_ONE);
        //
        optimizer.setInitLbd(-0.1);
        optimizer.setInitUbd(0.1);
        //
        // Obligatory things...
        //
        optimizer.setRnd(new Random(1234));
        optimizer.setParameters(f.arity());
        optimizer.updateObjective(f);
        //
        // for observing the optimization process.
        //
        optimizer.addListener(new BasicOptimizationListener());
        //
        optimizer.initialize();
        //
        // go!
        //
        optimizer.iterate(10000, 0.0);
        //
        // read the best solution.
        //
        //final EchoStateNetwork esn_solution = new EchoStateNetwork(3, reservoirsize, 3);
        double[] esn_solution_weights = new double[f.arity()];
        optimizer.readBestSolution(esn_solution_weights, 0);
        //
        // calculate Wout
        //
        //
        for (int i = 0; i < (reservoirsize ) * (3+ add_bias); i++) {
            esn_solution_weights[i] = esn_solution_weights[i] * feedBackScaling;
        }
        //
        double[] oldWeights = new double[esn.getWeightsNum()];
        double[] newWeights = new double[esn.getWeightsNum()];
        esn.readWeights(oldWeights);
        double[] Wout = Arrays.copyOfRange(oldWeights, f.arity(), esn.getWeightsNum());
        //
        System.arraycopy(esn_solution_weights, 0, newWeights, 0, esn_solution_weights.length);
        System.arraycopy(Wout, 0, newWeights, esn_solution_weights.length, Wout.length);
        esn.writeWeights(newWeights);  // flatten? Arrays.stream(W_reservoir).flatMapToInt(Arrays::stream).toArray();
        double error = esn.trainESN(Sequence, washout, training, test);
        //
        //get Wout again
        //
        double[] finalWeights = new double[esn.getWeightsNum()];
        esn.readWeights(finalWeights);
        //
        // write weights to file
        //
        try{
            Serializer.write(finalWeights, "data/esn-3-" + reservoirsize + "-3.weights");
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("done");
    	
    }

    
    // *
	 //* Helper method for sequence loading from file.
	 //
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