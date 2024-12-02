import './header.css';
const background = `${process.env.PUBLIC_URL}/assets/images/background.png`;

function Header() {
    return (

        <div className="header">
            <div className="column">
                <h1 className="first_line">Welcome to the &nbsp;  LLMazing Contracts </h1>
            </div>
            <div className="column">
                <img
                    src={background}
                    alt="Background decoration"
                    className="backgroundImage"
                    loading="lazy"
                />
            </div>
        </div>

    );
}
export default Header;