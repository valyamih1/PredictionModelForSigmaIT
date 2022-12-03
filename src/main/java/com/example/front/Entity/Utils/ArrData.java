package com.example.front.Entity.Utils;

import com.example.front.Entity.DataForFigure;

import java.util.List;

public class ArrData {
    private List<DataForFigure> arr;

    public ArrData(List<DataForFigure> arr) {
        this.arr = arr;
    }

    public ArrData(){}

    public List<DataForFigure> getArr() {
        return arr;
    }

    public void setArr(List<DataForFigure> arr) {
        this.arr = arr;
    }
}
