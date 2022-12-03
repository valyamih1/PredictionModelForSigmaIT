package com.example.front.Entity;

public class DataForFigure {
    private String date;
    private double realV;
    private double predictV;

    public DataForFigure(String date, double realV, double predictV) {
        this.date = date;
        this.realV = realV;
        this.predictV = predictV;
    }

    public DataForFigure() {}

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public double getRealV() {
        return realV;
    }

    public void setRealV(double realV) {
        this.realV = realV;
    }

    public double getPredictV() {
        return predictV;
    }

    public void setPredictV(double predictV) {
        this.predictV = predictV;
    }
}
