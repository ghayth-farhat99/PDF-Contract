const orangeLogo = `${process.env.PUBLIC_URL}/assets/images/orange-logo.svg`;

function NavBar() {
    return (
        <div>
            <nav className="navbar navbar-expand-lg" data-bs-theme="dark">
                <div className="container-fluid">
                    <div className="navbar-brand">
                        <a className="stretched-link" href="/">
                            <img
                                src={orangeLogo}
                                alt="Boosted Logo - Back to Home"
                                loading="lazy"
                                style={{ width: '50px', height: '50px' }}
                                className="logo-image" // Optional: add a class for styling
                            />
                        </a>
                        <h1 className="two-lined">
                            LLMazing
                            <br />
                            Contracts
                        </h1>
                    </div>
                    <button className="navbar-toggler ms-auto" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                        <span className="navbar-toggler-icon"></span>
                    </button>
                    <div className="collapse navbar-collapse" id="navbarNav">
                        <ul className="navbar-nav">
                            <li className="nav-item">
                                <a className="nav-link active" aria-current="page" href="/">Home</a>
                            </li>
                            <li className="nav-item">
                                <a className="nav-link" href="/">About</a>
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>
        </div>
    );
}
export default NavBar;