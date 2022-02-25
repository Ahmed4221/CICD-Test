from cgitb import reset
from helperfunctions import *

def main():
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    dataset = getData().copy()
    dataset = preprocessData(dataset)
    train_dataset,test_dataset = splitTraintTest(dataset)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop(TARGET_VARIABLE)
    test_labels = test_features.pop(TARGET_VARIABLE)
    normalizer = getNormalizer()
    normalizer.adapt(np.array(train_features))
    first = np.array(train_features[:1])
    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print()
        print('Normalized:', normalizer(first).numpy())
    
    dnn_model = build_and_compile_model(normalizer)
    print(dnn_model.summary())
    history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=100)
    test_results = {}
    test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
    predictions = predict(test_features,dnn_model)
    #plotting predictions
    plotResults(test_labels,predictions)
    #checking commit condition
    reset,previous_loss = gitReset(test_results)
    if not reset:
        #writeResults
        writeResults(test_results)


    #writing the results to the metrics file for better visualization
    with open('metrics.txt','w') as outfile:
        outfile.write("Performance of DNN is : {}".format(str(test_results['dnn_model'])))
        if reset:
            outfile.write("\n The commit was reset for not reaching threshold")
            outfile.write("\n Previous => {} VS current => {} ".format(str(previous_loss),str(test_results['dnn_model'])))

    

if __name__ == '__main__':
    main()
