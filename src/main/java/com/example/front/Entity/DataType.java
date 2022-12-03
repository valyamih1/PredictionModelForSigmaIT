package com.example.front.Entity;

public enum DataType {
    DAY("Day"),
    WEEK("Week"),
    MONTH("Month"),
    YEAR("Year");

    private String name;
    private DataType(String name){
        this.name = name;
    }

    public String getName(){
        return name;
    }
}

