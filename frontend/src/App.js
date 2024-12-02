import './App.css';
import 'boosted/dist/css/boosted.min.css';
import 'boosted/dist/js/boosted.bundle.min.js';
import NavBar from './components/navbar/navbar';
import Main from './pages/main/main';

function App() {
  return (
    <div className="App">
     
    <NavBar/>
      
    <Main/>

    </div>
  );
}

export default App;
