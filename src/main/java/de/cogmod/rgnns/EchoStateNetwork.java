package de.cogmod.rgnns;

import static de.cogmod.rgnns.ReservoirTools.*;

/**
 * @author Sebastian Otte
 */
public class EchoStateNetwork extends RecurrentNeuralNetwork {

    private double[][] inputweights;
    private double[][] reservoirweights;
    private double[][] outputweights;
    
    public double[][] getInputWeights() {
        return this.inputweights;
    }
    
    public double[][] getReservoirWeights() {
        return this.reservoirweights;
    }
    
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
        // Do I train over the same sequence?
        // How long should the teacherForcing training be? - Row size of M
        // Sequence = Number of Sequences x Squence Length ? Muss ich 1000 Sequences aufnehmen?
        // How to write Weights (Seralizer), map double[][] to double[]
        // How to keep information after Training run or does M expand to inf - Baches?
        // Why copy ESN in generateESNFutureProjection before unrolling prediction?
        // => To have one Network in actual state of the simulation, while the other predicts into the future
        // ?=> How does the ESN no old locations - it has to be washed in bevor prediction, teacher forcer all the time!
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

        int training_sequence_length = 200;
        for (int trainingstep = 0; trainingstep < training; trainingstep++) {
            System.out.println("washout with teacher forcing");
            //
            int t;
            for (t=0; t < washout; t++) {

                //
                // The semantic here is as follows:
                //
                // o We assume that the current target value
                //   is the desired output of the current
                //   calculation step of the ESN -> thus,
                //   we want that ideally output is the
                //   same as target.
                //
                // o Via teacher forcing,
                //   we enforce the ESN to unfold its
                //   dynamics as it would have produced
                //   target as output.
                //
                this.teacherForcing(sequence[t]);

                //
                // Perform an oscillator forward pass
                // in which the esn is fed with its
                // previous output (note that during
                // teacher forcing the output of the ESN
                // is overwritten with the target)
                //
                double[] output = this.forwardPassOscillator();
                //System.out.println(output);
            }
            double[][] M = new double[training_sequence_length][this.getLastInputLength()];
            for (int step = 0; step < training_sequence_length; step++){
                this.teacherForcing(sequence[t+step]);
                double[] output = this.forwardPassOscillator();
                M[step] = output;
            }
            final double[][] W_out   = new double[this.outputweights.length][this.outputweights[0].length];
            solveSVD(M, sequence, W_out);
            //this.writeWeights(W_out);
        }
        return 0.0; // error.
    }
    
}

// Generelle Fragen
// Nur non-linear im Reservour - keine gelernten non-linearities?
// -> Kann man so auch sowas wie einen Sinus mit exponentiell wachsender Amplitude modellieren?
//