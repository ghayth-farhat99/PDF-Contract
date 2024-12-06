import React from 'react';
import './main.css';
import useSteps from '../../hooks/useSteps';
import Step0 from '../../components/steps/step0';
import Step1 from '../../components/steps/step1';
import Step2 from '../../components/steps/step2';
import Step3 from '../../components/steps/step3';

function Main() {
    const totalSteps = 4;
    const initialMessage = "";

    // Destructure all necessary state and methods from the useSteps hook
    const {
        currentStep,
        nextStep,
        prevStep,
        message,
        setMessage,
        messageUnitTest,
        setMessageUnitTest,
        messageSmartContract,
        setMessageSmartContract,
        OnChangeFile,    // Step 0 file handler
        uploadFile,       // Step 0 upload handler
        loading,
        downloadHardhat,
        optionAnalyse,
        setOptionAnalyse,
        startAnalyse,
        solcResult,
        slitherResult,
        hardhatResult,
        eslintResult,
        startEslint,
        RegenerateUnitTestBasedOnEslint,
        correct
    } = useSteps(initialMessage, 0, totalSteps);

    const renderStep = () => {
        switch (currentStep) {
            case 0:
                return (
                    <Step0
                        onNext={uploadFile}
                        OnChangeFile={OnChangeFile}
                        loading={loading}
                    />
                );
            case 1:
                return (
                    <Step1
                        onNext={nextStep}
                        onBack={prevStep}
                        message={message}
                        setMessage={setMessage}
                        loading={loading}
                    />
                );
            case 2:
                return (
                    <Step2
                        onNext={nextStep}
                        onBack={prevStep}
                        messageUnitTest={messageUnitTest}
                        setMessageUnitTest={setMessageUnitTest}
                        loading={loading}
                        startEslint={startEslint}
                        eslintResult={eslintResult}
                        RegenerateUnitTestBasedOnEslint={RegenerateUnitTestBasedOnEslint}
                    />
                );
            case 3:
                return (
                    <Step3
                        onNext={nextStep}
                        onBack={prevStep}
                        messageSmartContract={messageSmartContract}
                        setMessageSmartContract={setMessageSmartContract}
                        loading={loading}
                        optionAnalyse={optionAnalyse}
                        setOptionAnalyse={setOptionAnalyse}
                        startAnalyse={startAnalyse}
                        solcResult={solcResult}
                        hardhatResult={hardhatResult}
                        slitherResult={slitherResult}
                        messageUnitTest={messageUnitTest}
                        setMessageUnitTest={setMessageUnitTest}
                        correct={correct}
                        download={downloadHardhat}
                    />
                );
            default:
                return null;
        }
    };

    return (
        <div className="main">
            {renderStep()}
        </div>
    );
}

export default Main;

