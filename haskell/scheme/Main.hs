module Main where
-- for getArgs
import System.Environment
-- for liftM
import Control.Monad
-- for error handling
import Control.Monad.Error

import Text.ParserCombinators.Parsec hiding (spaces)
import System.Environment

main :: IO ()
main = do
  putStrLn "Hello"
  -- args <- getArgs
  -- putStrLn ("Hello, " ++ args !! 0)
  -- (expr:_) <- getArgs
  -- putStrLn (readExpr "$")


myadd :: Num a => a -> a -> a
myadd a b = a + b

symbol :: Parser Char
symbol = oneOf "!#$%&|*+-/:<=>?@^_~"

readExpr :: String -> LispVal
readExpr input = case parse
  -- (spaces >> symbol)
  parseExpr
  "lisp" input of
                   -- the parse function returns an Either, it
                   -- contains two data constructors, Left for error,
                   -- Right for success and gives value
                   Left err -> String $ "No match: " ++ show err
                   Right val -> val

spaces :: Parser ()
spaces = skipMany1 space

data LispVal = Atom String
             | List [LispVal]
             | DottedList [LispVal] LispVal
             | Number Integer
             | String String
             | Bool Bool

-- used return to lift a String (one of the LispVal constructor) into
-- the Parser monad
parseString :: Parser LispVal
parseString = do
  -- _ <- is used to suppress the warning of discard a value of do
  -- notation
  _ <- char '"'
  -- rules to use >>, >>=, do
  -- if no passing values, use >>
  -- if parsing value to next expression, use >>=
  -- otherwise use "do" to bind
  x <- many (noneOf "\"")
  _ <- char '"'
  -- $ is a infix function applicatoin
  -- it is equivalent to return (String x)
  -- it has right associcativity, low precedence
  return $ String x

parseAtom :: Parser LispVal
parseAtom = do
  -- <|> is in Parsec, a combinator semantically, it tries one by one,
  -- until one succeeds. But there's one requirement: it must be LL(1),
  -- i.e. it cannot consume. Parsec allows backtracking, via "try"
  first <- letter <|> symbol
  rest <- many (letter <|> digit <|> symbol)
  -- Connect first and rest into a list, using cons operator (:).
  -- This is equivalent to using [first] ++ rest.  Note that first is
  -- not a list, thus it must be bracketed.
  let atom = first:rest
  return $ case atom of
    "#t" -> Bool True
    "#f" -> Bool False
    _ -> Atom atom

parseNumber :: Parser LispVal
-- many1 is a Parsec combinator that matches one or more, it returns a
-- string. We first use read to convert it into a number, then use
-- Number constructor to construct a LispVal.
--
-- the . here is a function composition operator. It creates a
-- function that first apply right function, then apply left function.
--
-- However, many1 will not return a raw string, but a Parser
-- String. So applying liftM on the function (Number . read) will let
-- it able to consume the String inside the monad.
-- liftM is inside Control.Monad
parseNumber = liftM (Number . read) $ many1 digit

-- using liftM is not that beautiful, so consider these alternatives
-- alternative 1
parseNumber1 :: Parser LispVal
parseNumber1 = do x <- many1 digit
                  (return . Number . read) x
-- alternative 2
parseNumber2 :: Parser LispVal
parseNumber2 = many1 digit >>= \x -> (return . Number . read) x


parseExpr :: Parser LispVal
parseExpr = parseAtom
            <|> parseString
            <|> parseNumber
            <|> parseQuoted
            <|> do char '('
                   -- try here is to allow backtracking.  It tries one
                   -- by one, and if it fails, it backs up the
                   -- previous state
                   x <- try parseList <|> parseDottedList
                   char ')'
                   return x

parseList :: Parser LispVal
parseList = liftM List $ sepBy parseExpr spaces

parseDottedList :: Parser LispVal
parseDottedList = do
  head1 <- endBy parseExpr spaces
  tail1 <- char '.' >> spaces >> parseExpr
  return $ DottedList head1 tail1

parseQuoted :: Parser LispVal
parseQuoted = do
  char '\''
  x <- parseExpr
  return $ List [Atom "quote", x]




-- evaluation

showVal :: LispVal -> String
showVal (String contents) = "\"" ++ contents ++ "\""
showVal (Atom name) = name
showVal (Number contents) = show contents
showVal (Bool True) = "#t"
showVal (Bool False) = "#f"
showVal (List contents) = "(" ++ unwordsList contents ++ ")"
showVal (DottedList head tail) = "(" ++ unwordsList head
                                 ++ " . " ++ showVal tail
                                 ++ ")"

unwordsList :: [LispVal] -> String
-- point-free style: writing definitions purely in terms of function
-- composition and partial application
-- unwords is a Haskell function that add spaces between list items
unwordsList = unwords . map showVal

-- support show on LispVal
instance Show LispVal where show = showVal


eval :: LispVal -> LispVal
eval val@(String _) = val
eval val@(Number _) = val
eval val@(Bool _) = val
eval (List [Atom "quote", val]) = val

eval (List (Atom func : args)) = apply func $ map eval args

apply :: String -> [LispVal] -> LispVal
-- lookup is a Haskell function that looks up a key inside a list of
-- pairs. It returns a Maybe.
apply func args = maybe (Bool False) ($ args) $ lookup func primitives

primitives :: [(String, [LispVal] -> LispVal)]
primitives = [("+", numericBinop (+)),
              ("-", numericBinop (-)),
              ("*", numericBinop (*)),
              ("/", numericBinop div),
              ("mod", numericBinop mod),
              ("quotient", numericBinop quot),
              ("remainder", numericBinop rem)]

-- takes a primitive haskell function
-- wrap it with code to unpack an argument list
-- apply the function to the arguments
-- wrap result in Number constructor
--
-- But why the op has (Int -> Int -> Int)? This can support arbitrary
-- operants??
numericBinop :: (Integer -> Integer -> Integer) -> [LispVal] -> LispVal
-- foldl1 is said to implement arbitrary operants support
numericBinop op params = Number $ foldl1 op $ map unpackNum params
unpackNum :: LispVal -> Integer
unpackNum (Number n) = n
-- reads is a Haskell function that parse a string into a pair (parsed
-- number, rest string)
unpackNum (String n) = let parsed = reads n :: [(Integer, String)] in
  if null parsed
     then 0
     -- !! means index 0. Then why need fst??
     else fst $ parsed !! 0
-- match one-item list
unpackNum (List [n]) = unpackNum n
-- otherwise
unpackNum _ = 0



-- Error handling

data LispError = NumArgs Integer [LispVal]
               | TypeMismatch String LispVal
               | Parser ParseError
               | BadSpecialForm String LispVal
               | NotFunction String String
               | UnboundVar String String
               | Default String
showError :: LispError -> String
showError (UnboundVar message varname)
  = message ++ ": " ++ varname
showError (BadSpecialForm message form)
  = message ++ ": " ++ show form
showError (NotFunction message func)
  = message ++ ": " ++ show func
showError (NumArgs expected found)
  = "Expected " ++ show expected
  ++ " args; found values " ++ unwordsList found
showError (TypeMismatch expected found)
  = "Invalid type: expected " ++ expected
  ++ ", found " ++ show found
showError (Parser parseErr)
  = "Parse error at " ++ show parseErr

instance Show LispError where show = showError

-- Make it an instance of the builtin Error class.  This is necessary
-- for GHC's built-in error handling functions.
instance Error LispError where
  noMsg = Default "An error has occurred"
  strMsg = Default

-- This is partial evaluation. The rule for type acts same as
-- functions.
type ThrowsError = Either LispError
