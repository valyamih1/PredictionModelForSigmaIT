package com.example.front.Entity.Utils;

public class RawData {
    private String t;
    private String pay;
    private String predict;

    public RawData(String date, String realV, String predictV) {
        this.t = date;
        this.pay = realV;
        this.predict = predictV;
    }

    public RawData() {
    }

    public String getT() {
        return t;
    }

    public void setT(String t) {
        this.t = t;
    }

    public String getRealV() {
        return pay;
    }

    public void setRealV(String realV) {
        this.pay = realV;
    }

    public String getPredict() {
        return predict;
    }

    public void setPredict(String predict) {
        this.predict = predict;
    }
}
