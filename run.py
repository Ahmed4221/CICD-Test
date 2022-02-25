from helperfunctions import *


def main():
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
    performance = pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T
    predictions = predict(test_features,dnn_model)

    # #plotting predictions
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.savefig('loss.png')

    #writing the results to the metrics file for better visualization
    with open('metrics.txt','w') as outfile:
        outfile.write(str(performance.iloc[0]))
    
    assert test_results['dnn_model']>1 != 0,"Accuracy threshold not reached"
if __name__ == '__main__':
    main()
