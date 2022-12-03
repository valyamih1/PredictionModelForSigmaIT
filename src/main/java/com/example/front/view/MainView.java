package com.example.front.view;


import com.example.front.Entity.DataForFigure;
import com.example.front.Entity.DataType;
import com.example.front.Service.DataService;
import com.vaadin.flow.component.charts.Chart;
import com.vaadin.flow.component.charts.model.*;
import com.vaadin.flow.component.combobox.ComboBox;
import com.vaadin.flow.component.html.Div;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.component.progressbar.ProgressBar;
import com.vaadin.flow.router.Route;

import org.springframework.web.bind.annotation.CrossOrigin;

import java.util.Collections;
import java.util.List;

@CrossOrigin(origins = "*", allowedHeaders = "*")
@Route("")
public class MainView extends VerticalLayout{

    private ComboBox<DataType> comboBox = new ComboBox<>("Data type filter");
    private Chart chart = new Chart();
    private DataService dataService;

    ProgressBar progressBar = new ProgressBar();
    Div progressBarLabel = new Div();


    public MainView(DataService dataService) {
        this.dataService = dataService;
        comboBox.setItems(DataType.values());
        comboBox.setItemLabelGenerator(DataType::getName);
        comboBox.addValueChangeListener(e -> getNewForecast());

        progressBar.setIndeterminate(true);
        progressBarLabel.setText("Generating report...");
        progressBar.setVisible(true);
        progressBarLabel.setVisible(true);
        add(comboBox,chart,progressBar,progressBarLabel);
        getChart(Collections.EMPTY_LIST, null);
    }

    private void getNewForecast() {
        DataType value = comboBox.getValue();
        if (value == null) {
            initLoad(true);
            getChart(Collections.EMPTY_LIST, null);
        } else {
            List<DataForFigure> data = dataService.getData(value.getName());
            initLoad(false);
            getChart(data, value.getName());
        }
    }

    private void initLoad(boolean val) {
        progressBar.setVisible(val);
        progressBarLabel.setVisible(val);
    }


    public void getChart(List<DataForFigure> data,String name){
        remove(chart);
        chart = new Chart();
        Configuration conf = chart.getConfiguration();
        conf.getChart().setType(ChartType.LINE);

        if (data.size()>0) {
            DataSeries serie1 = new DataSeries("real");
            DataSeries serie2 = new DataSeries("predict");
            for (int i=0;i<data.size();i++){
                serie1.add(new DataSeriesItem(data.get(i).getDate(),data.get(i).getRealV()));
                serie2.add(new DataSeriesItem(data.get(i).getDate(),data.get(i).getPredictV()));
            }
            conf.addSeries(serie1);
            conf.addSeries(serie2);

            conf.setTitle("LINES");
            XAxis xAxis = conf.getxAxis();
            xAxis.setTitle("Pay_Date");
            xAxis.setType(AxisType.DATETIME);
            YAxis yAxis = conf.getyAxis();
            yAxis.setTitle("Pay");

            Legend legend = conf.getLegend();
            legend.getTitle().setText("Income amount");
        }else{
            DataSeries serie1 = new DataSeries("real");
            DataSeries serie2 = new DataSeries("predict");
            conf.setTitle("Don't have data");
            conf.addSeries(serie1);
            conf.addSeries(serie2);
        }
        add(chart);
    }

}
