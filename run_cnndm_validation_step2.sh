export CLASSPATH=[path-to-stanford-corenlp-3.9.2.jar-or-more-recent-versions]

for f in ./cnn_dm/*.hypo  # use appropriate directory depending on the task
do 
    echo "Processing $f..."

    # Tokenize
    cat $f | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $f.tokenized
    cat ./cnn_dm/test.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ./cnn_dm/test.hypo.target

    # Compute rouge
    files2rouge $f.tokenized ./cnn_dm/test.hypo.target > $f.result.out
done

