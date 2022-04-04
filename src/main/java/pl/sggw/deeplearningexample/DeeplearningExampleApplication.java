package pl.sggw.deeplearningexample;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import pl.sggw.deeplearningexample.deeplearning4j.IrisClassifier;

import java.io.IOException;

@SpringBootApplication
public class DeeplearningExampleApplication {

	public static void main(String[] args) throws IOException, InterruptedException {
		SpringApplication.run(DeeplearningExampleApplication.class, args);
		IrisClassifier classifier = new IrisClassifier();
		classifier.execute();
	}

}
