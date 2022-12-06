package com.example.front.Service;

import com.example.front.Entity.DataForFigure;
import com.example.front.Entity.Utils.RawData;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.lang.reflect.Type;
import java.net.SocketTimeoutException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
public class DataService {
    @Value("${URL_PYTHON_SERVER}")
    private String URL_SERVER;

    public List<DataForFigure> getData(String name){
        OkHttpClient client = new OkHttpClient().newBuilder().build();
        Request request = new Request.Builder().
                url("http://"+URL_SERVER+"/getData?date="+name)
                .get()
                .build();
        Response response = null;
        try {
            response = client.newCall(request).execute();

            Gson gson = new Gson();
            Type listType = new TypeToken<List<RawData>>(){}.getType();
            List<RawData> arr = gson.fromJson(response.body().string(), listType);

            List<DataForFigure> result = new ArrayList<>();
            DataForFigure tmpValue;
            String sDate;
            String realV;
            String predictV;
            for (int i=0;i<arr.size()-1;i++) {
                tmpValue = new DataForFigure();
                sDate = arr.get(i).getT().split(" ")[0];
                String[] split = sDate.split("-");
                tmpValue.setDate(split[2]+"-"+split[1]+"-"+split[0]);
                realV = arr.get(i).getRealV();
                if (!realV.equals("nan")) {
                    tmpValue.setRealV(Double.parseDouble(realV));
                    predictV = arr.get(i).getPredict();
                    if (!predictV.equals("nan")) {
                        tmpValue.setPredictV(Double.parseDouble(predictV));
                    }
                    result.add(tmpValue);
                }
            }
            return result;

        }
        catch (SocketTimeoutException e){
            return Collections.EMPTY_LIST;
        }
        catch (IOException e) {
            e.printStackTrace();
            return Collections.EMPTY_LIST;
        }
    }
}
