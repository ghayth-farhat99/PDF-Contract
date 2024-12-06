import { useState } from 'react';
const useSteps = (initialMessage, initialStep = 0, totalSteps) => {
    const [currentStep, setCurrentStep] = useState(initialStep);
    const [message, setMessage] = useState(initialMessage);
    const [messageUnitTest, setMessageUnitTest] = useState("");
    const [messageSmartContract, setMessageSmartContract] = useState("");
    const [inputFile, setInputFile] = useState(null); // For Step 0 file input
    const [loading, setLoading] = useState(false); // Loading state
    const [optionAnalyse, setOptionAnalyse] = useState([])
    const [solcResult, setSolcResult] = useState("");
    const [slitherResult, setSlitherResult] = useState("");
    const [hardhatResult, setHardhatResult] = useState("");
    const [eslintResult, setEslintResult] = useState("");
    // Helper function to make fetch requests
    const fetchData = async (url, bodyContent, setterFunction) => {
        try {
            setLoading(true); // Start loading
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(bodyContent)
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const jsonResponse = await response.json();
            setterFunction(jsonResponse);
        } catch (error) {
            console.error(`Error in ${url} request:`, error);
        } finally {
            setLoading(false); // Stop loading regardless of success or failure
        }
    };

    // Step 1 - File upload logic
    const OnChangeFile = (e) => {
        setInputFile(e.target.files[0]);
        console.log("File selected:", e.target.files[0]);
    };

    const saveMessageUnitTest = async () => {

        try {
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/save_unit_test`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ code: messageUnitTest }),
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const result = await response.json();
            console.log("Unit test saved successfully:", result);
        } catch (error) {
            console.error("Failed to save unit test:", error);
        }
    }

    const saveMessageSmartContract = async () => {

        try {
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/save_smart_contract`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ code: messageSmartContract }),
            });

            if (!response.ok) {
                throw new Error(`Error: ${response.statusText}`);
            }

            const result = await response.json();
            console.log("Unit test saved successfully:", result);
        } catch (error) {
            console.error("Failed to save unit test:", error);
        }
    }

    const startAnalyse = async () => {
        try {
            setLoading(true); // Start loading
            await saveMessageSmartContract()
            await saveMessageUnitTest()
            // Fetch the results for each tool only if the checkbox is checked
            if (optionAnalyse.includes('solcCheck')) {
                const solcResponse = await fetch(`${process.env.REACT_APP_BACKEND_URL}/analyze_contract_solc`, { method: 'POST' });
                const solcData = await solcResponse.json();
                await setSolcResult(Object.entries(solcData.compilation_step)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join("\n"));
            }

            if (optionAnalyse.includes('slitherCheck')) {
                const slitherResponse = await fetch(`${process.env.REACT_APP_BACKEND_URL}/analyze_contract_slither`, { method: 'POST' });
                const slitherData = await slitherResponse.json();
                await setSlitherResult(Object.entries(slitherData.slither_response)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join("\n"));
            }

            if (optionAnalyse.includes('hardhatCheck')) {
                const hardhatResponse = await fetch(`${process.env.REACT_APP_BACKEND_URL}/analyze_contract_hardhat`, { method: 'POST' });
                const hardhatData = await hardhatResponse.json();
                await setHardhatResult(Object.entries(hardhatData.hardhat_response)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join("\n"));
            }

        } catch (error) {
            console.error('Error during analysis:', error);
        } finally {
            setLoading(false); // Stop loading regardless of success or failure
        }
    };

    const uploadFile = async (e) => {
        e.preventDefault();

        if (!inputFile) {
            console.error("No file selected for upload.");
            return;
        }

        // Create FormData and append the file
        const formData = new FormData();
        formData.append("file", inputFile);

        try {
            setLoading(true); // Start loading for file upload
            const response = await fetch(
                `${process.env.REACT_APP_BACKEND_URL}/upload`,
                {
                    method: "POST",
                    body: formData,
                });

            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }

            const data = await response.json();
            console.log('File uploaded successfully:', data);
            setMessage(data.response); // Save the response from the upload
            nextStep(); // Move to the next step

        } catch (error) {
            console.error('Error uploading file:', error);
        } finally {
            setLoading(false); // Stop loading when done
        }
    };

    const downloadHardhat = async () => {
        try {
            setLoading(true); // Start loading
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/download`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'hardhat_test_env.zip';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Error downloading the file:', error);
        } finally {
            setLoading(false); // Stop loading regardless of success or failure
        }
    };

    const startEslint = async () => {
        try {
            await saveMessageUnitTest()
            setLoading(true);
            const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/analyze_test`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            await setEslintResult(Object.entries(data.eslint_response)
                .map(([key, value]) => `${key}: ${value}`)
                .join("\n"));
        } catch (error) {
            console.error('Error starting the eslint:', error);
        } finally {
            setLoading(false); // Stop loading regardless of success or failure
        }
    };

    const RegenerateUnitTestBasedOnEslint = async () => {
        try{
            setLoading(true);
            const response = await fetchData(
                `${process.env.REACT_APP_BACKEND_URL}/regenerate_unit_test_eslint`,
                { eslint_results: eslintResult},
                (data) => setMessageUnitTest(data["unit_tests"])
            );
            console.log(response);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }

    const correct = async (solc_status, slither_status, hardhat_status) => {
        if(solc_status === 'failed'){
            RegenerateSmartContractBasedOnSolc();
        }else{
            if(hardhat_status === 'failed'){
                RegenerateSmartContractAndUnitTestBasedOnHardhat(); 
            }else{
                RegenerateSmartContractBasedOnSlither();
            }
        }
    }
    const RegenerateSmartContractBasedOnSolc = async () => {
        try{
            setLoading(true);
            const response = await fetchData(
                `${process.env.REACT_APP_BACKEND_URL}/regenerate_smart_contract_solc`,
                { solc_results: solcResult},
                (data) => setMessageSmartContract(data["smart_contract"])
            );
            console.log(response);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }

    const RegenerateSmartContractBasedOnSlither = async () => {
        try{
            setLoading(true);
            const response = await fetchData(
                `${process.env.REACT_APP_BACKEND_URL}/regenerate_smart_contract_slither`,
                { slither_results: slitherResult},
                (data) => setMessageSmartContract(data["smart_contract"])
            );
            console.log(response);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }

    const RegenerateSmartContractAndUnitTestBasedOnHardhat = async () => {
        try {
            console.log("hardhatResult :", hardhatResult);
            setLoading(true);
            const response = await fetchData(
                `${process.env.REACT_APP_BACKEND_URL}/regenerate_smart_contract_hardhat`,
                { hardhat_results: hardhatResult },
                (data) => {
                    setMessageSmartContract(data["smart_contract"]);
                    setMessageUnitTest(data["unit_test"]); // Fix: Use "unit_test" as per your endpoint response
                }
            );
            console.log(response);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };
    

    const nextStep = async () => {
        switch (currentStep) {
            case 1:
                await fetchData(
                    `${process.env.REACT_APP_BACKEND_URL}/generate_unit_test`,
                    { extract_informations: message },
                    (data) => setMessageUnitTest(data["unit_tests"])

                );
                break;

            case 2:
                await fetchData(
                    `${process.env.REACT_APP_BACKEND_URL}/generate_smart_contract`,
                    { unit_tests: messageUnitTest },
                    (data) => setMessageSmartContract(data["smart_contract"])
                );
                await saveMessageUnitTest()
                await setOptionAnalyse(["solcCheck", "slitherCheck", "hardhatCheck"])
                await startAnalyse()
                await setOptionAnalyse([])
                break;

            default:
                break;
        }

        setCurrentStep((prev) => Math.min(prev + 1, totalSteps - 1));
    };

    const prevStep = () => {
        console.log("currentStep : ", currentStep)
        if(currentStep === 3){
            setSolcResult("");
            setSlitherResult("");
            setHardhatResult("");
        }else{
            if(currentStep === 2){
                setEslintResult("");
            }
        }
        setCurrentStep((prev) => Math.max(prev - 1, 0));
    };

    const reset = () => {
        setCurrentStep(initialStep);
        setMessage(initialMessage);
        setMessageUnitTest("");
        setMessageSmartContract("");
        setEslintResult("");
        setInputFile(null);
    };

    return {
        currentStep,
        nextStep,
        prevStep,
        reset,
        message,
        setMessage,
        messageUnitTest,
        setMessageUnitTest,
        /*         messagePrompt,
                setMessagePrompt, */
        messageSmartContract,
        setMessageSmartContract,
        OnChangeFile,
        uploadFile,
        inputFile,
        loading, // Return loading state
        downloadHardhat,// download the smart contract .sol in the browser
        optionAnalyse,
        setOptionAnalyse,
        startAnalyse,
        solcResult,
        slitherResult,
        hardhatResult,
        eslintResult, // results from eslint tool
        startEslint,
        RegenerateUnitTestBasedOnEslint,
        correct
    };

};

export default useSteps;
