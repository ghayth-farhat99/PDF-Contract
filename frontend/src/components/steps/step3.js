import React, { useState, useRef, useEffect } from 'react';
import Settings from '../settings/settings';

const Step3 = ({ onBack, onNext, messageUnitTest, setMessageUnitTest, messageSmartContract, setMessageSmartContract, loading, optionAnalyse, setOptionAnalyse, startAnalyse, solcResult, slitherResult, hardhatResult, download }) => {
    const [activeTab, setActiveTab] = useState('smartContract');
    const [activeTabBottom, setActiveTabBottom] = useState('solc'); // Bottom tabs: Solc, Slither, Hardhat
    const handleCheckboxChange = (event) => {
        const { id, checked } = event.target;

        // Add or remove the option from the 'optionAnalyse' array based on checkbox state
        setOptionAnalyse((prevState) => {
            if (checked) {
                return [...prevState, id];  // Add to array if checked
            } else {
                return prevState.filter((item) => item !== id);  // Remove from array if unchecked
            }
        });
    };

    // Function to handle Smart Contract download
    const downloadSmartContract = () => {
        const blob = new Blob([messageSmartContract], { type: 'text/plain' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'MyContract.sol';
        link.click();
        URL.revokeObjectURL(link.href);
    };

    const [showToast, setShowToast] = useState(false); // State to control toast visibility
    const [showToastContent, setShowToastContent] = useState(''); // State to control toast visibility

    const [toastMessage, setToastMessage] = useState(''); // Toast message state

    const codeAreaRef = useRef(null);
    const lineNumberRef = useRef(null);

    const [scrollTop, setScrollTop] = useState(0);

    // Function to handle scroll events and sync scroll position
    const handleScroll = (e) => {
        setScrollTop(e.target.scrollTop); // Update scrollTop when the user scrolls
    };

    // Function to generate line numbers
    const generateLineNumbers = (code) => {
        const lines = code.split('\n');
        return lines.map((_, index) => index + 1);
    };

    // Sync scrollTop for both the code area and line numbers
    useEffect(() => {
        if (codeAreaRef.current) {
            codeAreaRef.current.scrollTop = scrollTop;
        }
        if (lineNumberRef.current) {
            lineNumberRef.current.scrollTop = scrollTop;
        }
    }, [scrollTop]);

    const handleCopy = (code, title) => {
        navigator.clipboard.writeText(code).then(() => {
            setToastMessage('Code copied to clipboard!');
            setShowToast(true);
            setShowToastContent(title)
            setTimeout(() => setShowToast(false), 3000); // Hide toast after 3 seconds
        }).catch((err) => {
            console.error('Failed to copy: ', err);
        });
    };

    return (
        <div className="fifth-step">
            <Settings />
            <p className="first_paragraph">Transform PDF Contracts into Smart Contracts</p>
            <div className="container_step">
                <nav className="stepped-process" aria-label="Progress navigation">
                    <ol className="stepped-process-list">
                        <li className="stepped-process-item">
                            <span className="stepped-process-link">Extract information</span>
                        </li>
                        <li className="stepped-process-item">
                            <span className="stepped-process-link">Generate unit test</span>
                        </li>
                        <li className="stepped-process-item active">
                            <span className="stepped-process-link">Generate and Analyse the smart contract</span>
                        </li>
                    </ol>
                </nav>
                <div class='container-flex'>
                    <div style={{ width: '150%', margin: 0 }} className="card text-center custom-card">
                        <div className="card-header text-body bg-body">
                            <ul className="nav nav-pills card-header-pills">
                                <li className="nav-item">
                                    <button
                                        className={`nav-link ${activeTab === 'smartContract' ? 'active' : ''}`}
                                        onClick={() => setActiveTab('smartContract')}
                                    >
                                        Smart Contract Generated
                                    </button>
                                </li>
                                <li className="nav-item">
                                    <button
                                        className={`nav-link ${activeTab === 'unitTest' ? 'active' : ''}`}
                                        onClick={() => setActiveTab('unitTest')}
                                    >
                                        Unit Test Generated
                                    </button>
                                </li>
                            </ul>
                        </div>

                        <div className="card-body">
                            {activeTab === 'smartContract' && (
                                <div id="smartContract">
                                    <h5 className="card-title">Smart Contract Code</h5>
                                    <div className="d-flex" style={{ position: 'relative', width: '100%' }}>
                                        <button
                                            onClick={() => handleCopy(messageSmartContract, 'Smart Contract Code Copied')}
                                            type="button"
                                            className="btn btn-primary position-absolute top-0 end-0"
                                            style={{
                                                marginTop: '1.5vh',
                                                marginRight: '1.5vw',
                                                '--bs-btn-padding-y': '.25rem',
                                                '--bs-btn-padding-x': '.5rem',
                                                '--bs-btn-font-size': '.75rem',
                                            }}
                                        >
                                            Copy
                                        </button>

                                        <div className="d-flex" style={{ width: '100%' }}>
                                            {/* Line numbers column */}
                                            <div
                                                ref={lineNumberRef}
                                                className="form-control bg-dark text-white"
                                                style={{
                                                    width: '50px',
                                                    userSelect: 'none',
                                                    height: '296px',
                                                    flexDirection: 'column',
                                                    overflowY: 'hidden',
                                                }}
                                            >
                                                {generateLineNumbers(messageUnitTest).map((lineNumber) => (
                                                    <div key={lineNumber} style={{ flex: '0 0 auto' }}>{lineNumber}</div>
                                                ))}
                                            </div>

                                            {/* Code area column */}
                                            <textarea
                                                ref={codeAreaRef}
                                                className="form-control bg-dark text-white "
                                                rows="10"
                                                value={messageSmartContract}
                                                onChange={(e) => {
                                                    setMessageSmartContract(e.target.value)
                                                }
                                                }
                                                onScroll={handleScroll}
                                                style={{
                                                    fontSize: '16px',
                                                    marginLeft: '0px',
                                                    width: 'calc(100% - 40px)', // Account for line number width
                                                    height: '300px',
                                                    paddingLeft: '10px',
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            )}

                            {activeTab === 'unitTest' && (
                                <div id="unitTest">
                                    <h5 className="card-title">Unit Test Code</h5>
                                    <div className="d-flex" style={{ position: 'relative', width: '100%' }}>
                                        <button
                                            onClick={() => handleCopy(messageUnitTest, 'Unit Test Code Copied')}
                                            type="button"
                                            className="btn btn-primary position-absolute top-0 end-0"
                                            style={{
                                                marginTop: '1.5vh',
                                                marginRight: '1.5vw',
                                                '--bs-btn-padding-y': '.25rem',
                                                '--bs-btn-padding-x': '.5rem',
                                                '--bs-btn-font-size': '.75rem',
                                            }}
                                        >
                                            Copy
                                        </button>

                                        <div className="d-flex" style={{ width: '100%' }}>
                                            {/* Line numbers column */}
                                            <div
                                                ref={lineNumberRef}
                                                className="form-control bg-dark text-white"
                                                style={{
                                                    width: '50px',
                                                    userSelect: 'none',
                                                    height: '296px',
                                                    flexDirection: 'column',
                                                    overflowY: 'hidden',
                                                }}
                                            >
                                                {generateLineNumbers(messageUnitTest).map((lineNumber) => (
                                                    <div key={lineNumber} style={{ flex: '0 0 auto' }}>{lineNumber}</div>
                                                ))}
                                            </div>

                                            {/* Code area column */}
                                            <textarea
                                                ref={codeAreaRef}
                                                className="form-control bg-dark text-white "
                                                rows="10"
                                                value={messageUnitTest}
                                                onChange={(e) => setMessageUnitTest(e.target.value)}
                                                onScroll={handleScroll}
                                                style={{
                                                    fontSize: '16px',
                                                    marginLeft: '0px',
                                                    width: 'calc(100% - 40px)', // Account for line number width
                                                    height: '300px',
                                                    paddingLeft: '10px',
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    <div 
                        style={{ 
                            display: 'flex', 
                            flexDirection: 'column', 
                            justifyContent: 'center', 
                            alignItems: 'center', 
                            margin: '5px', 
                            padding: '10px', 
                            height: 'auto', 
                            maxWidth: '90%', 
                            width: '300px', // Fallback for smaller sizes
                            boxSizing: 'border-box'
                        }} 
                        className="card text-center custom-card"
                    >
                        <div style={{ width: '100%', marginTop: '7%' }} className="form-check">
                            <input
                                checked={optionAnalyse.includes('solcCheck')}
                                onChange={handleCheckboxChange}
                                className="form-check-input" 
                                type="checkbox" 
                                value="" 
                                id="solcCheck" 
                            />
                            <label style={{ display: 'block', textAlign: 'left' }} className="form-check-label" htmlFor="solcCheck">
                                Solc Analyse
                            </label>
                        </div>
                        <div style={{ width: '100%' }} className="form-check">
                            <input
                                checked={optionAnalyse.includes('slitherCheck')}
                                onChange={handleCheckboxChange}
                                className="form-check-input" 
                                type="checkbox" 
                                value="" 
                                id="slitherCheck" 
                            />
                            <label style={{ display: 'block', textAlign: 'left' }} className="form-check-label" htmlFor="slitherCheck">
                                Slither Analyse
                            </label>
                        </div>
                        <div style={{ width: '100%' }} className="form-check">
                            <input
                                checked={optionAnalyse.includes('hardhatCheck')}
                                onChange={handleCheckboxChange}
                                className="form-check-input" 
                                type="checkbox" 
                                value="" 
                                id="hardhatCheck" 
                            />
                            <label style={{ display: 'block', textAlign: 'left' }} className="form-check-label" htmlFor="hardhatCheck">
                                Hardhat Analyse
                            </label>
                        </div>
                        <button
                            onClick={startAnalyse}
                            style={{ 
                                marginTop: '10px', 
                                width: '100%', 
                                maxWidth: '200px', 
                                alignSelf: 'center' 
                            }}
                            type="button"
                            className="btn btn-primary btn-sm"
                            id="startAnalyseButton"
                        >
                            Start Analyse
                        </button>
                    </div>

                </div>

                {/* Second Box - Terminal Style */}
                <div style={{ width: '105%', margin: 0 }} className="card text-center custom-card mt-3">
                    <div className="card-header text-body bg-body">
                        <ul className="nav nav-pills card-header-pills">
                            <li className="nav-item">
                                <button
                                    className={`nav-link ${activeTabBottom === 'solc' ? 'active' : ''}`}
                                    onClick={() => setActiveTabBottom('solc')}
                                >
                                    Solc Result
                                </button>
                            </li>
                            <li className="nav-item">
                                <button
                                    className={`nav-link ${activeTabBottom === 'slither' ? 'active' : ''}`}
                                    onClick={() => setActiveTabBottom('slither')}
                                >
                                    Slither Result
                                </button>
                            </li>
                            <li className="nav-item">
                                <button
                                    className={`nav-link ${activeTabBottom === 'hardhat' ? 'active' : ''}`}
                                    onClick={() => setActiveTabBottom('hardhat')}
                                >
                                    Hardhat Result
                                </button>
                            </li>
                        </ul>
                    </div>
                    <div className="card-body">
                        <div
                            style={{
                                backgroundColor: '#1e1e1e',
                                color: '#d4d4d4',
                                fontFamily: 'monospace',
                                padding: '1rem',
                                borderRadius: '5px',
                                overflowY: 'auto',
                                maxHeight: '300px',
                                textAlign: 'left',
                            }}
                        >
                            {activeTabBottom === 'solc' && <pre>{solcResult}</pre>}
                            {activeTabBottom === 'slither' && <pre>{slitherResult}</pre>}
                            {activeTabBottom === 'hardhat' && <pre>{hardhatResult}</pre>}
                        </div>
                    </div>
                </div>

                {loading && (
                    <div style={{ marginBottom: "2vh" }} className="loading-indicator">
                        <div className="spinner-border text-primary" role="status">
                            <span className="visually-hidden">Loading...</span>
                        </div>
                    </div>
                )}

                <div className="button-group" 
                 style={{
                    display: 'flex',
                    justifyContent: 'flex-end',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '10px',
                    padding: '10px',
                    maxWidth: '100%',
                    boxSizing: 'border-box',
                  }}
                  >
                <button type="button" className="btn btn-outline-secondary" onClick={onBack}>
                    Back
                </button>

                <button type="button" className="btn btn-secondary" onClick={downloadSmartContract}>
                    Download Smart Contract
                </button>

                <button type="button" className="btn btn-secondary" onClick={download}>
                    Download Hardhat Project ZIP
                </button>
                </div>


                {/* Custom Toast Notification */}
                {showToast && (
                    <div className="toast-container position-fixed bottom-0 end-0 ">
                        <div className="toast show" role="alert" aria-live="assertive" aria-atomic="true">
                            <div className="toast-header">
                                <strong className="me-auto">{showToastContent}</strong>
                                <small>Just Now</small>
                                <button type="button" className="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                            </div>
                            <div className="toast-body">
                                {toastMessage}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Step3;
