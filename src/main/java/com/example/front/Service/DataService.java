package com.example.front.Service;

import com.example.front.Entity.DataForFigure;
import com.example.front.Entity.Utils.ArrData;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.net.SocketTimeoutException;
import java.util.Collections;
import java.util.List;

@Service
public class DataService {
    @Value("${URL_PYTHON_SERVER}")
    private String URL_SERVER;



    public List<DataForFigure> getData(String name){
        OkHttpClient client = new OkHttpClient().newBuilder().build();
        Request request = new Request.Builder().
                url("http://"+URL_SERVER+"/getData")
                .get()
                .build();
        Response response = null;
        try {
            response = client.newCall(request).execute();
            //ObjectMapper objectMapper = new ObjectMapper();
            //JsonNode jsonNode = objectMapper.readTree(response.body().string());
            //String loc = jsonNode.get("region_name").asText();
            /*Gson gson = new Gson();
            ArrData arrData = gson.fromJson(response.body().string(), ArrData.class);

            List<DataForFigure> result = new ArrayList<>();
            arrData.getArr().forEach(e->result.add(e));
            return result;*/
            System.out.println("Sleep");
            Thread.sleep(2000);
            System.out.println("Unsleep");
            String s ="1";
            return Collections.EMPTY_LIST;
        }
        catch (SocketTimeoutException e){
            return Collections.EMPTY_LIST;
        }
        catch (IOException e) {
            e.printStackTrace();
            return Collections.EMPTY_LIST;
        } catch (InterruptedException e) {
            e.printStackTrace();
            return Collections.EMPTY_LIST;
        }
    }
}
