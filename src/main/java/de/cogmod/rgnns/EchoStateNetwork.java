package de.cogmod.rgnns;

import java.sql.Array;
import java.util.Random;
import java.util.Arrays;
import static de.cogmod.rgnns.ReservoirTools.*;

/**
 * @author Sebastian Otte
 */
public class EchoStateNetwork extends RecurrentNeuralNetwork {

    private double[][] inputweights;
    private double[][] reservoirweights;
    private double[][] outputweights;

    public int reservoirsize;
    
    public double[][] getInputWeights() {
        return this.inputweights;
    }
    
    public double[][] getReservoirWeights() {
        return this.reservoirweights;
    }

    public double[] getReservoirActivations;
    
    public double[][] getOutputWeights() {
        return this.outputweights;
    }
    
    public EchoStateNetwork(
        final int input,
        final int reservoirsize,
        final int output
    ) {
        super(input, reservoirsize, output);
        //
        this.inputweights     = this.getWeights()[0][1];
        this.reservoirweights = this.getWeights()[1][1];
        this.outputweights    = this.getWeights()[1][2];
        this.reservoirsize = reservoirsize;
        //
    }
    
    @Override
    public void rebufferOnDemand(int sequencelength) {
        super.rebufferOnDemand(1);
    }
    

    public double[] output() {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer();
        //
        final int n = act[outputlayer].length;
        //
        final double[] result = new double[n];
        final int t = Math.max(0, this.getLastInputLength() - 1);
        //
        for (int i = 0; i < n; i++) {
            result[i] = act[outputlayer][i][t];
        }
        //
        return result;
    }

    public double[] getReservoirActivations() {
        //
        final int bias = 1;
        final double[][][] act = this.getAct();
        final int reservoirLayer  = this.getOutputLayer() - 1;
        //
        final int n = act[reservoirLayer].length + bias;
        //
        final double[] result = new double[n];
        final int t = Math.max(0, this.getLastInputLength() - 1);
        //
        for (int i = 0; i < n - bias; i++) {
            result[i] = act[reservoirLayer][i][t];
        }
        if(bias == 1) {
            result[n - 1] = 0;
        }

        //
        return result;
    }
    
    /**
     * This is an ESN specific forward pass realizing 
     * an oscillator by means of an output feedback via
     * the input layer. This method requires that the input
     * layer size matches the output layer size. 
     */
    public double[] forwardPassOscillator() {
        //
        // this method causes an additional copy operation
        // but it is more readable from outside.
        //
        final double[] output = this.output();
        return this.forwardPass(output);
    }
    
    /**
     * Overwrites the current output with the given target.
     */
    public void teacherForcing(final double[] target) {
        //
        final double[][][] act = this.getAct();
        final int outputlayer  = this.getOutputLayer(); 
        //
        final int n = act[outputlayer].length;
        //
        final int t = this.getLastInputLength() - 1;
        //
        for (int i = 0; i < n; i++) {
            act[outputlayer][i][t] = target[i];
        }
    }
    
    /**
     * ESN training algorithm. 
     */
    public double trainESN(
        final double[][] sequence,
        final int washout,
        final int training,
        final int test
    ) {
        double[][] train_seq = Arrays.copyOfRange(sequence,washout,washout+training);
        double[][] test_seq = Arrays.copyOfRange(sequence,washout+training,washout+training+test);

        System.out.println("washout with teacher forcing");

        // Washout Run
        int t;
        for (t=0; t < washout; t++) {
            double[] output = this.forwardPassOscillator();
            this.teacherForcing(sequence[t]);
        }

        // Training Run
        double[][] M = new double[training][this.reservoirsize + 1]; // +1 für Bias oder nicht
        for (int step = 0; step < training; step++){
            double[] output = this.forwardPassOscillator();
            M[step] = this.getReservoirActivations();
            this.teacherForcing(sequence[washout+step]);
        }
        // Calculate new Weight
        solveSVD(M, train_seq, this.getOutputWeights()); // Null spalte für Bias ist dabei, könnte man löschen, vlt wird der optimizer confused, ich bins :D


        // Test Run
        double[][] M_test = new double[test][this.getLastInputLength()];
        this.teacherForcing(sequence[washout+training]);
        for (int step = 0; step < test; step++){
            double[] output = this.forwardPassOscillator();
            M_test[step] = output;
        }

        // calculate MSE
        double error = RMSE(M_test,test_seq);

        return error; // error.
    }

    
}

// Generelle Fragen
// Nur non-linear im Reservour - keine gelernten non-linearities?
// -> Kann man so auch sowas wie einen Sinus mit exponentiell wachsender Amplitude modellieren?
// Tutorial fuer VAE  : towards science varational auto encoder code tutorial


// Do I train over the same sequence?
// How long should the teacherForcing training be? - Row size of M
// Sequence = Number of Sequences x Squence Length ? Muss ich 1000 Sequences aufnehmen?
// How to write Weights (Seralizer), map double[][] to double[]
// How to keep information after Training run or does M expand to inf - Baches?
// Why copy ESN in generateESNFutureProjection before unrolling prediction?
// => To have one Network in actual state of the simulation, while the other predicts into the future
// ?=> How does the ESN no old locations - it has to be washed in before prediction, teacher forcer all the time!
// Have *random* weights in the realm of 10^-8 for Input Weights? And scale W0 with eigenvalue

        /*
        // Just checking W_out
        System.out.println("W_out rows");
        System.out.println(this.outputweights.length);
        System.out.println("W_out cols");
        System.out.println(this.outputweights[0].length);
        System.out.println("W_out :");
        System.out.println(matrixAsString(this.outputweights,2));
        */

//for (int trainingstep = 0; trainingstep < training; trainingstep++) {

//error
        /*
        double error = 0;
        for (int i=0; i< test; i++){
            error = error + (comp_seq[0][i] - test_seq[0][i]) * (comp_seq[0][i] - test_seq[0][i]);
        }
*/