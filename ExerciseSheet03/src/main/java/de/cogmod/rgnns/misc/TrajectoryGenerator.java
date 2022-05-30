package de.cogmod.rgnns.misc;

public interface TrajectoryGenerator {
    public void reset();
    public int vectorsize();
    public double[] next();
}