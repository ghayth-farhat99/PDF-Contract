import React from 'react';
import Header from '../header/header';
import Settings from '../settings/settings'

const Step0 = ({ OnChangeFile, onNext, loading }) => {
    return (
        <div className='first-step'>
            <Header/>
            <Settings/>
            <p className="first_paragraph">Transform PDF Contracts into Smart Contracts</p>
            <div className="upload_container">
                <div className="input-group mb-3">
                    <input
                        type="file"
                        name="file"
                        className="form-control"
                        id="inputGroupFile01"
                        onChange={OnChangeFile}
                        required
                    />
                </div>
            </div>
            {loading && (
                <div style={{marginBottom:"2vh"}} className="loading-indicator">
                    <div className="spinner-border text-primary" role="status">
                        <span className="visually-hidden">Loading...</span>
                    </div>
                </div>
            )}
            <div className="btn-group" role="group" aria-label="Second group">
                <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={onNext}
                    disabled={loading}
                >
                    {loading ? 'Generating...' : 'Next'}
                </button>
            </div>
        </div>
    );
};

export default Step0;
