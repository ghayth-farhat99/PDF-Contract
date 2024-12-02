import globals from "globals";
import pluginJs from "@eslint/js";

/** @type {import('eslint').Linter.Config[]} */
export default [
  // Match JavaScript files
  { 
    files: ["**/*.js"], 
    languageOptions: { 
      sourceType: "commonjs", 
      globals: {
        ...globals.node, // Include Node.js globals
        ...globals.mocha // Include Mocha globals like 'describe', 'it', 'beforeEach'
      }
    }
  },
  // Apply the recommended JavaScript rules
  pluginJs.configs.recommended,
  // Rules to customize the linting behavior
  {
    rules: {
      "no-unused-vars": ["warn"], // Warn instead of error for unused variables
      "no-undef": "error",          // Disable no-undef for testing globals
    }
  }
];
