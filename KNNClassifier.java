import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class KNNClassifier {

	static int K;    // k in kNN Algorithm
	Hashtable<Integer, String>document_labels=new Hashtable<Integer, String>(); //data structure to hold document class labels
	static LinkedHashMap<Integer, Hashtable<String, Integer>> documents=new LinkedHashMap<Integer, Hashtable<String, Integer>>(); //holds all the documents and store terms present in each document along with their frequency count as <1, <<122,3>, <11,1>,<113,1>,<24,2>>> e.g. 1 is document number and it contains term 24 two times.
	static ArrayList<Integer> docKeys=new ArrayList<Integer>(); //store all document number as keys
	static HashMap<Integer, Double> accuracyKUnWeighted=new HashMap<Integer, Double>(); //stores accuracies for different values of k as <k, accuracy> in unweighted kNN
	static HashMap<Integer, Double> accuracyKWeighted=new HashMap<Integer, Double>(); //stores accuracies for different values of k as <k, accuracy> in weighted kNN
	static final String validationCommand = "validation"; //validation command parameter
	static final String weightedCommand = "weighted"; //weighted command parameter
	static final String KCommand = "k";   //k command parameter
	static String validationType="cross"; //validation type with value as cross. This is default value for it.
	static boolean weighted=true; //weighted value as true.This is default value for it.
	static final String cross ="cross"; //cross validation type
	static final String split ="split"; //split validation type

	public static void main(String[] args) {
		KNNClassifier knnClassifier=new KNNClassifier();    //instance of KNNClassifier class
		HashMap<String, String>  arguments=new HashMap<String, String>();
		for(int i=0;i<args.length;i++)
		{
			try
			{
				arguments.put(args[i].split("=")[0].toLowerCase(), args[i].split("=")[1].toLowerCase()); //if command line arguments are given by user store them as key value pairs like <parameter, value> e.g. <validation, cross> or <k, 7>
			}
			catch(Exception e)
			{
				System.out.println("Invalid command !");   //if command is given and its given in an incorrect way, do not proceed
				System.exit(-1);
			}
		}
		if(args.length>0)    // if any argument or parameter is specified by the user
		{
			try
			{
				if(arguments.get(KCommand)==null && arguments.get(validationCommand)==null &&   //if some invalid command parameters are given other than k, validation and weighted
						arguments.get(weightedCommand)==null)
				{
					{
						System.out.println("Invalid command parameters !");
						return;
					}
				}

				if(arguments.get(KCommand)!=null)
				{
					K=Integer.parseInt(arguments.get(KCommand));   //get the value of k specified by the user
					if(K==0)
					{
						System.out.println("Invalid K !");
						return;
					}
				}
				if(arguments.get(validationCommand)==null)    
				{
					validationType="cross";               // if validation type is not given in command, choose default as 10 fold cross validation.
				}
				else
				{
					validationType=arguments.get(validationCommand); //get the user specified validation type
					if(!validationType.equalsIgnoreCase(cross) && !validationType.equalsIgnoreCase(split))
					{
						System.out.println("Invalid validation option!");  //if user mentioned incorrect value for validation type
						return;
					}
				}
				if(arguments.get(weightedCommand)==null)    //if weighted parameter is not specified default will be weighted kNN 
				{
					weighted=true;
				}
				else
				{
					weighted=Boolean.parseBoolean(arguments.get(weightedCommand));  //get the user specified value of weighted parameter
				}
				knnClassifier.load_document_labels(); //read the document label file and populate "document_labels" data structure  
				knnClassifier.load_corpus(); //read the news article matrix file and populate "documents" data structure
				for(Map.Entry<Integer, Hashtable<String, Integer>> doc : documents.entrySet())
				{
					docKeys.add(doc.getKey());  //populate 'dockeys' with all documents as keys
				}
				Collections.shuffle(docKeys);       //randomize or shuffle all the keys to mix them well. This is done to remove any sort of bias or pattern if present in the matrix data.
				if(arguments.get(KCommand)!=null)
					System.out.println("k="+K);
				if(!weighted)
					System.out.println("unweigted kNN");
				else
					System.out.println("weigted kNN");	
				if(validationType.equalsIgnoreCase(split))     //if validation type is split evaluate model using 70/30 split.
				{
					System.out.println("validation=70/30 % split");
					if(arguments.get(KCommand)!=null)            // if k value is specified in the command, calculate accuracy using split for that value of k
					{											//use weighted parameter if specified by the user else default is true (i.e run for weighted kNN)
						knnClassifier.evaluate_using_split();
					}
					else
					{
						accuracyKUnWeighted.clear();        //if k is not specified by the user, then find accuracy for k= 1 to 10 using split
						accuracyKWeighted.clear();
						for(int i=1;i<=10;i++)
						{
							K=i;		
							knnClassifier.evaluate_using_split();
						}
						knnClassifier.showHighestKAccuracy();  //show the k value for which highest accuracy was achieved 
					}
				}
				else                          //if validation type is cross evaluate model using 10 fold cross validation.
				{

					System.out.println("validation=10 fold cross validation");  //if validation type is not specified by the user, default is always 10 fold cross validation
					if(arguments.get(KCommand)!=null)
					{
						knnClassifier.cross_validation(); // if k value is specified in the command, calculate accuracy using 10 fold cross validation for that value of k
					}
					else
					{
						accuracyKUnWeighted.clear();
						accuracyKWeighted.clear();
						for(int i=1;i<=10;i++)        //if k is not specified by the user, then find accuracy for k= 1 to 10 using 10 fold cross validation
						{
							K=i;
							knnClassifier.cross_validation();
						}
						knnClassifier.showHighestKAccuracy(); //show the k value for which highest accuracy was achieved 
					}
				}

			}
			catch (Exception e) {
				System.out.println("Invalid command !");  //if command is not correctly given, the program terminates
				System.exit(-1);
			}
		}
		else    // if no parameter at all specified by the user then run both weighted and unweighted kNN for k=1 to 10 using 10 fold cross validation.                                     
		{
			knnClassifier.load_document_labels();  //load labels
			knnClassifier.load_corpus();   //load matrix data
			for(Map.Entry<Integer, Hashtable<String, Integer>> doc : documents.entrySet())
			{
				docKeys.add(doc.getKey());   //store document as keys in a list
			}
			Collections.shuffle(docKeys);  //shuffle keys to remove any bias or pattern in the data.
			accuracyKUnWeighted.clear();
			System.out.println("validation=10 fold cross validation");
			for(int i=1;i<=10;i++)               //unweighted kNN
			{
				K=i;
				weighted = false;
				knnClassifier.cross_validation();   //run 10 fold cross validation for k = 1 to 10 with unweighted kNN
			}
			knnClassifier.showHighestKAccuracy(); //show the k value for which highest accuracy was achieved with unweighted kNN
			accuracyKWeighted.clear();
			for(int i=1;i<=10;i++)               //weighted kNN
			{
				K=i;
				weighted = true;
				knnClassifier.cross_validation(); //run 10 fold cross validation for k = 1 to 10 with weighted kNN
			}
			knnClassifier.showHighestKAccuracy(); //show the k value for which highest accuracy was achieved with weighted kNN
		}

	}

	void evaluate_using_split()  //evaluate accuracy using 70/30 % split                   
	{
		long startTime = System.currentTimeMillis();
		System.out.println("Computing accuracy for (k="+K+")....");
		int document_count=document_labels.size();
		int train_data_size= (document_count*70)/100;   //find training data size which is 70% of all the documents
		HashMap<Integer,Hashtable<String, Integer>>test_corpus=new HashMap<Integer, Hashtable<String, Integer>>();  //holds 30% of documents as test corpus
		HashMap<Integer,Hashtable<String, Integer>>training_corpus=new HashMap<Integer, Hashtable<String, Integer>>(); //holds 70% of documents as training corpus
		int count=0; boolean flag=false; //flag and count are used to ensure that train data contains documents equal to 70% of the corpus which is train_data_size
		DecimalFormat dff=new DecimalFormat(".####");
		for(Integer key: docKeys)            //divide the documents as test and training corpus
		{
			if(!flag)
			{	
				count++;
				training_corpus.put(key, documents.get(key));  
				if(count==train_data_size)                    //store documents in the training corpus till 70% of the documents are stored
					flag=true;
			}
			else
			{
				test_corpus.put(key, documents.get(key));  //store remaining 30% in the test corpus
			}
		}
		double overall_accuracy=getAccuracy(training_corpus, test_corpus,weighted)*100; //find accuracy of the model for generated test and training data
		if(!weighted)          //If unweighted print its accuracy and store the corresponding k value
		{
			System.out.println("Unweighted kNN Accuracy(k="+K+"):"+dff.format(overall_accuracy)+"%");
			accuracyKUnWeighted.put(K, overall_accuracy);
		}
		else
		{                       //If weighted print its accuracy and store the corresponding k value
			System.out.println("weighted kNN Accuracy(k="+K+"):"+dff.format(overall_accuracy)+"%");
			accuracyKWeighted.put(K, overall_accuracy);
		}
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		System.out.println("Time:"+elapsedTime/1000.0+"secs");   //the total time it takes for the algorithm to compute accuracy for given k
	}

	void showHighestKAccuracy()   // function to find k for which highest accuracy achieved for k= 1 to 10
	{
		HashMap<Integer, Double> accuracyKScores=new HashMap<Integer, Double>();  //data structure to hold k value and corresponding accuracy values
		DecimalFormat dff=new DecimalFormat(".####");   //show accuracy upto 4 places of decimal
		if(!weighted)              
		{
			accuracyKScores = accuracyKUnWeighted;
		}
		else
		{
			accuracyKScores = accuracyKWeighted;
		}
		List<Entry<Integer, Double>> sortedAccuracy = new ArrayList<Entry<Integer, Double>>(accuracyKScores.entrySet());

		Collections.sort(sortedAccuracy,                        //sort accuracykScores on the basis of accuracy values. It is sorted in decreasing order.
				new Comparator<Entry<Integer, Double>>() {
			@Override
			public int compare(Entry<Integer, Double> e1, Entry<Integer, Double> e2) {
				return e2.getValue().compareTo(e1.getValue());
			}
		}
				);
		if(!weighted)
		{
			System.out.println("unweighted kNN highest Accuracy:"+dff.format(sortedAccuracy.get(0).getValue())+"% for K="+sortedAccuracy.get(0).getKey()); //display highest accuracy along with corresponding k value(when k varies between 1 to 10) for unweighted kNN
		}
		else
		{
			System.out.println("weighted kNN highest Accuracy:"+dff.format(sortedAccuracy.get(0).getValue())+"% for K="+sortedAccuracy.get(0).getKey()); //display highest accuracy along with corresponding k value(when k varies between 1 to 10) for weighted kNN
		}
	}
	void cross_validation()		//10-fold cross validation
	{
		long startTime = System.currentTimeMillis();
		System.out.println("Computing accuracy for (k="+K+")....");
		int document_count=document_labels.size();
		long test_data_size= Math.round((double)document_count/10); // split the documents into 10 folds. Calculate size of each fold.
		HashMap<Integer,Hashtable<String, Integer>>test_corpus=new HashMap<Integer, Hashtable<String, Integer>>();  //hold test data
		HashMap<Integer,Hashtable<String, Integer>>temp_corpus=new HashMap<Integer, Hashtable<String, Integer>>();  // hold data which is already used as a test data
		HashMap<Integer,Hashtable<String, Integer>>training_corpus=new HashMap<Integer, Hashtable<String, Integer>>(); //hold training data
		double accuracy=0;
		DecimalFormat dff=new DecimalFormat(".####");
		for(int i=1;i<=10;i++)     // for 10 folds
		{
			test_corpus.clear();
			training_corpus.clear();
			int count=0; boolean flag=false;   //flag and count are used to ensure that test fold contains documents equal the size of the fold which is test_data_size
			for(Integer key: docKeys)
			{
				if(!flag)
				{	
					if(temp_corpus.get(key)==null)   //if the document is not part this data structure make it to go in the test fold because its not used ever as test
					{
						count++;
						test_corpus.put(key, documents.get(key));          //store document into test fold 
						temp_corpus.put(key, documents.get(key));          //store document in this data structure to ensure that it does not become part of the test data again 
						if(i<10 && count==test_data_size)                 //if for the last fold there are still more than test_data_size entries left, then take all into test even though it is greater than test size because we are on the last fold and we can't divide further 
						{
							flag=true;                                    // hence don't make flag true for last fold. take in test whatever remains.
						}
					}
					else
					{
						training_corpus.put(key, documents.get(key));   //if the document is already used as test, it will be part of training data
					}
				}
				else
				{
					training_corpus.put(key, documents.get(key)); //if the test fold is full, the document will go into training fold
				}
			}
			accuracy=accuracy+getAccuracy(training_corpus, test_corpus,weighted);  //calculate accuracy for each fold
		}
		double overall_accuracy = (accuracy/10.0)*100;  //average all the 10 accuracies to get the overall accuracy of 10 fold cross validation
		//display overall accuracy for weighted or unweighted kNN
		if(!weighted)
		{
			System.out.println("Unweighted kNN Accuracy(k="+K+"):"+dff.format(overall_accuracy)+"%");
			accuracyKUnWeighted.put(K, overall_accuracy);
		}
		else
		{
			System.out.println("weighted kNN Accuracy(k="+K+"):"+dff.format(overall_accuracy)+"%");
			accuracyKWeighted.put(K, overall_accuracy);
		}
		long stopTime = System.currentTimeMillis();
		long elapsedTime = stopTime - startTime;
		System.out.println("Time:"+elapsedTime/1000.0+"secs");   //display the time used to compute accuracy in 10 fold cross validation
	}
	void load_corpus()  //populate news matrix into "documents" hashmap with keys as documents and values as hashtable of terms and their frequencies. This value hashtable contains keys as terms and their frequency as values. we can call it as term vector
	{
		System.out.println("Loading corpus....");
		try (BufferedReader br = new BufferedReader(new FileReader("news_articles.mtx"))) {  //read the matrix
			String line;
			int count=0;
			while ((line = br.readLine()) != null) { //read each line
				count++;
				if(count>2)
				{
					String content[]=line.split(" ");  //split into three parts => document, term and its frequency
					int document = Integer.parseInt(content[0].trim()); //fetch document 
					Hashtable<String, Integer>termVector;  //data structure to create term vector for each document
					if(documents.get(document)==null)      //if this the new document, create its new term vector
					{
						termVector=new Hashtable<String, Integer>();
					}
					else
					{
						termVector=documents.get(document); //if this the document is already encountered, fetch its term vector
					}
					termVector.put(content[1].trim(), Integer.parseInt(content[2].trim())); //update the term vector with the new term and its frequency for a given document
					documents.put(document, termVector);  //update the document for the new term
				}
			}
		}
		catch (Exception e) {
			System.out.println("Problem in loading corpus!");   //if data is corrupted, corpus can't be loaded
			System.exit(-1);
		}

	}

	void load_document_labels()         //read the labels from news_articles.labels
	{
		try (BufferedReader br = new BufferedReader(new FileReader("news_articles.labels"))) {
			String line;
			while ((line = br.readLine()) != null) {		
				String content[]=line.split(",");
				document_labels.put(Integer.parseInt(content[0].trim()), content[1].trim());  //store true labels/class for each document
			}
		}
		catch (Exception e) {
			System.out.println("Problem in loading document labels!");
			System.exit(-1);
		}
	}
	double getAccuracy(HashMap<Integer, Hashtable<String, Integer>> training_corpus,
			HashMap<Integer, Hashtable<String, Integer>> test_corpus,boolean weighted) //calculate accuracy for given training and test data
	{
		double correct=0;  //used to determine the correctly labeled documents
		for (Map.Entry<Integer, Hashtable<String, Integer>> doc : test_corpus.entrySet()) { //for each test document(unseen data) predict its class
			String predicted_class= kNN(training_corpus, doc,weighted);  //predict the class using kNN
			String actual_class= document_labels.get(doc.getKey());    //get the true class of the document
			if(predicted_class.equalsIgnoreCase(actual_class))   //if the both classes match, increase the correctly predicted document count
				correct++;
		}
		double accuracy= correct/(double)test_corpus.size();    //find proportion of correctly labeled documents in the training set
		return accuracy;
	}

	String kNN(HashMap<Integer, Hashtable<String, Integer>> training_corpus,
			Map.Entry<Integer, Hashtable<String, Integer>> newDoc, boolean weighted)  //kNN algorithm
	{
		Hashtable<Integer,Double> cosines=new Hashtable<Integer, Double>();  //used to store cosine value for each document 
		for (Map.Entry<Integer, Hashtable<String, Integer>> doc : training_corpus.entrySet()) {
			Hashtable<String, Integer>termVector=doc.getValue();   //for each document in training set, get its term vector
			Hashtable<String, Integer>newDoctermVector=newDoc.getValue(); //for each document in the test set, get its term vector
			double dotProductSum=0; //used to find the sum of the dot product of the two vectors
			for(Map.Entry<String, Integer> term : termVector.entrySet()) //find the numerator part of the cosine similarity
			{
				if(newDoctermVector.get(term.getKey())!=null)  // for each term of the training term vector, if it is present in the test term vector, multiply their frequencies
				{
					dotProductSum = dotProductSum + (newDoctermVector.get(term.getKey())*term.getValue());  //sum all the multiplied frequencies
				}

			}
			double newDocVectorNorm=0;
			double docVectorNorm=0;
			for(Map.Entry<String, Integer> newTerm : newDoctermVector.entrySet()) //find the denominator part of the cosine similarity
			{
				newDocVectorNorm=newDocVectorNorm+ (newTerm.getValue()*newTerm.getValue());  //get the sum of squares of term frequencies in a term vector of a test document
			}
			for(Map.Entry<String, Integer> term : termVector.entrySet())
			{
				docVectorNorm=docVectorNorm+ (term.getValue()*term.getValue()); //get the sum of squares of term frequencies in a term vector of a training document
			}
			double cosine = dotProductSum/Math.sqrt(docVectorNorm*newDocVectorNorm);  //find cosine similarity by diving both numerator and denominator
			cosines.put(doc.getKey(),cosine);  //store this cosine similarity value for each document
		}
		List<Entry<Integer,Double>> sortedCosines = new ArrayList<Entry<Integer,Double>>(cosines.entrySet());

		Collections.sort(sortedCosines,                    //sort the cosine similarity values in the decreasing order
				new Comparator<Entry<Integer,Double>>() {
			@Override
			public int compare(Entry<Integer,Double> e1, Entry<Integer,Double> e2) {
				return e2.getValue().compareTo(e1.getValue());
			}
		}
				);
		String predicted_class="";
		if(!weighted)
		{
			predicted_class=unweightedkNN(sortedCosines);   //find the predicted class based on sorted cosines using unweighted kNN
		}
		else
		{
			predicted_class=weightedkNN(sortedCosines);  //find the predicted class based on sorted cosines using weighted kNN
		}
		return predicted_class;
	}

	String unweightedkNN(List<Entry<Integer,Double>> sortedCosines)  //this function returns predicted class based on cosine values using voting method
	{
		HashMap<String, Integer>voting=new HashMap<String, Integer>();   //stores vote count for each class label
		for(int i=0;i<K;i++)   //select first k (k as in kNN) sorted cosines
		{
			if(i<sortedCosines.size())    //this check is introduced to ensure that if k in kNN is so large that it exceeds the size of documents in training set 
			{
				Entry<Integer,Double> doc=sortedCosines.get(i);    //fetch the documents in the decreasing order of cosine values
				String class_label=document_labels.get(doc.getKey());  //get the true label for each document
				if(voting.get(class_label)==null)   //if class label not present in the data structure, initialize it with vote 1 for the first time
				{
					voting.put(class_label, 1);
				}
				else
				{
					int votes=voting.get(class_label);  //if class label already exists, update its vote count by adding 1
					votes++;
					voting.put(class_label, votes); //store the new vote count for the class label
				}
			}
		}
		List<Entry<String,Integer>> sortedVotes = new ArrayList<Entry<String,Integer>>(voting.entrySet());

		Collections.sort(sortedVotes,   //sort the class labels based on their vote counts
				new Comparator<Entry<String,Integer>>() {
			@Override
			public int compare(Entry<String,Integer> e1, Entry<String,Integer> e2) {
				return e2.getValue().compareTo(e1.getValue());
			}
		}
				);

		String predicted_class=sortedVotes.get(0).getKey(); //the class label with the highest votes is the first element in this list and it is the one we are looking for
		return predicted_class;
	}

	String weightedkNN(List<Entry<Integer,Double>> sortedCosines) //this function returns predicted class based on cosine values using weight kNN method
	{
		HashMap<String, Double>weights=new HashMap<String, Double>(); //stores  weight for each class label
		for(int i=0;i<K;i++) //select first k (k as in kNN) sorted cosines
		{
			if(i<sortedCosines.size())
			{
				Entry<Integer,Double> doc=sortedCosines.get(i);  //fetch the documents in the decreasing order of cosine values. 
				String class_label=document_labels.get(doc.getKey()); //get the true label for each document
				if(weights.get(class_label)==null)  
				{
					weights.put(class_label, doc.getValue()); //if class label not present in the data structure, initialize it with its value of weight for the first time
				}                                            //weights are taken as cosine similarity scores
				else
				{
					double weight=weights.get(class_label);    //if class label present in the data structure, fetch the current weight of the class label(similarity score) and add the new weight to it
					weight=weight+doc.getValue();
					weights.put(class_label, weight);   //here the weighting scheme is to sum all the cosine similarity scores of same class
				}
			}
		}
		List<Entry<String,Double>> sortedWeights = new ArrayList<Entry<String,Double>>(weights.entrySet());

		Collections.sort(sortedWeights,   //sort the weights in decreasing order
				new Comparator<Entry<String,Double>>() {
			@Override
			public int compare(Entry<String,Double> e1, Entry<String,Double> e2) {
				return e2.getValue().compareTo(e1.getValue());
			}
		}
				);

		String predicted_class=sortedWeights.get(0).getKey();  //the first element in the sorted list will have highest weight and it is the we are looking for
		return predicted_class;
	}
}
