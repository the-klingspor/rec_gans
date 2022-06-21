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

import static de.cogmod.rgnns.ReservoirTools.*;
import static de.cogmod.rgnns.ReservoirTools.matrixAsString;

public class DELearningESN {

    public static void main(String[] args) {

        final int reservoir_size = 30;
        final double feedBackScaling = 1e-8;
        //final double[][] W_input = new double[3][reservoir_size];
        //final double[][] W_reservoir = new double[reservoir_size][reservoir_size];
        final EchoStateNetwork esn = new EchoStateNetwork(3, reservoir_size, 3);
        int add_bias = 1;

        final Objective f = new Objective() {
            //
            @Override
            public int arity() {
                return (reservoir_size + add_bias)*(reservoir_size + add_bias) + (3 + add_bias)*(reservoir_size + add_bias); // Add Bias
            } // is W_reservoir + W_input
            @Override
            /**
             * This is the callback method that is called from the
             * optimizer to compute the "fitness" of a particular individual.
             */
            public double compute(double[] values, int offset) {
                double[][] Sequence = new double[3][];//loadSequence("some/file");  // pointer auf weights
                int washout = 100;
                int training = 1000;
                int test = 100;

                //double[] inputWeights = Arrays.copyOfRange(values,offset,(reservoir_size+add_bias)*3);
                //double[] reservoirWeights = Arrays.copyOfRange(values,offset,(reservoir_size+add_bias)*(reservoir_size+add_bias));

                double [] optimizedWeights = Arrays.copyOfRange(values,offset,arity());
                for (int i=0; i<(reservoir_size+add_bias)*3; i++){
                    optimizedWeights[i] = optimizedWeights[i]*feedBackScaling;
                }

                double[] oldWeights = new double[esn.getWeightsNum()];
                double[] newWeights = new double[esn.getWeightsNum()];
                esn.readWeights(oldWeights);
                double[] Wout = Arrays.copyOfRange(oldWeights,arity(),esn.getWeightsNum());

                System.arraycopy(optimizedWeights, 0, newWeights, 0, optimizedWeights.length);
                System.arraycopy(Wout, 0, newWeights, optimizedWeights.length, Wout.length);
                esn.writeWeights(newWeights);  // flatten? Arrays.stream(W_reservoir).flatMapToInt(Arrays::stream).toArray();

                double error =  esn.trainESN(Sequence,washout,training,test);
                return error;
            }
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
        optimizer.iterate(1000, 0.0);
        //
        // read the best solution.
        //
        final EchoStateNetwork esn_solution = new EchoStateNetwork(3, f.arity(), 3);
        //optimizer.readBestSolution(esn_solution, 0);
        //map(solution, 0, x);
        //
        // Print out solution. Note that least squares solution is:
        // 0.68
        // -0.07
        // -0.10
        // -0.05
        // 0.69
        //
        System.out.println();
    	
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