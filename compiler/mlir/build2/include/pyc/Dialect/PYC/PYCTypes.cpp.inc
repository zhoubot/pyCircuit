/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::pyc::ClockType,
::pyc::ResetType

#endif // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::pyc::ClockType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::pyc::ClockType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Case(::pyc::ResetType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::pyc::ResetType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::pyc::ClockType>([&](auto t) {
      printer << ::pyc::ClockType::getMnemonic();
      return ::mlir::success();
    })
    .Case<::pyc::ResetType>([&](auto t) {
      printer << ::pyc::ResetType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace pyc {


} // namespace pyc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::pyc::ClockType)
namespace pyc {


} // namespace pyc
MLIR_DEFINE_EXPLICIT_TYPE_ID(::pyc::ResetType)

#endif // GET_TYPEDEF_CLASSES

